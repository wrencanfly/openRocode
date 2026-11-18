//! Pluggable Control Plane & Tensor Plane abstractions.
//!
//! - v0.1: Unix Domain Socket (UDS) NDJSON subprocess channel with strict OS-level read timeouts
//!   for S1/L3 policy worker supervision + 64-slot `InMemoryTensorRingBuffer`.
//! - v0.3: Zero-copy POSIX Shared Memory Ring Buffer (`ShmRingBuffer` + Seqlock) & Zenoh bridge.

use std::fs;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rocode_proto::{FrameRef, PolicyWorkerConfig, RobotState, WorkerRequest, WorkerResponse};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("slot {0} expired or overwritten in tensor ring buffer")]
    SlotOverwritten(u32),
    #[error("worker heartbeat timed out (> {0} ms)")]
    HeartbeatTimeout(u64),
    #[error("worker subprocess crashed or closed socket: {0}")]
    WorkerCrashed(String),
    #[error("transport I/O error: {0}")]
    Io(String),
}

/// Zero-copy-ready abstraction for reading and publishing high-bandwidth sensor/tensor payloads.
pub trait TensorPlane: Send + Sync {
    fn publish_frame(
        &self,
        stream_id: &str,
        timestamp_ns: u64,
        payload: &[u8],
    ) -> Result<FrameRef, TransportError>;
    fn read_frame(&self, frame: &FrameRef) -> Result<Vec<u8>, TransportError>;
}

struct TensorSlot {
    seq: u64,
    stream_id: String,
    payload: Vec<u8>,
}

/// Fixed-capacity ring buffer preserving `FrameRef` overwrite semantics (v0.1 precursor to SHM Seqlock).
pub struct InMemoryTensorRingBuffer {
    capacity: usize,
    next_seq: AtomicU64,
    slots: Mutex<Vec<Option<TensorSlot>>>,
}

impl InMemoryTensorRingBuffer {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(4);
        Self {
            capacity: cap,
            next_seq: AtomicU64::new(1),
            slots: Mutex::new((0..cap).map(|_| None).collect()),
        }
    }
}

impl TensorPlane for InMemoryTensorRingBuffer {
    fn publish_frame(
        &self,
        stream_id: &str,
        timestamp_ns: u64,
        payload: &[u8],
    ) -> Result<FrameRef, TransportError> {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let slot_id = (seq as usize % self.capacity) as u32;
        let mut guard = self
            .slots
            .lock()
            .map_err(|_| TransportError::Io("mutex poisoned".into()))?;
        guard[slot_id as usize] = Some(TensorSlot {
            seq,
            stream_id: stream_id.to_string(),
            payload: payload.to_vec(),
        });
        Ok(FrameRef {
            stream_id: stream_id.to_string(),
            seq,
            timestamp_ns,
            slot_id,
        })
    }

    fn read_frame(&self, frame: &FrameRef) -> Result<Vec<u8>, TransportError> {
        let idx = frame.slot_id as usize;
        if idx >= self.capacity {
            return Err(TransportError::SlotOverwritten(frame.slot_id));
        }
        let guard = self
            .slots
            .lock()
            .map_err(|_| TransportError::Io("mutex poisoned".into()))?;
        match &guard[idx] {
            Some(slot) if slot.seq == frame.seq && slot.stream_id == frame.stream_id => {
                Ok(slot.payload.clone())
            }
            _ => Err(TransportError::SlotOverwritten(frame.slot_id)),
        }
    }
}

static SOCKET_SEQ: AtomicU64 = AtomicU64::new(1);

/// Isolated external policy worker process connected via Unix Domain Socket with strict timeouts.
pub struct UdsPolicySubprocess {
    socket_path: PathBuf,
    child: Child,
    writer: UnixStream,
    reader: BufReader<UnixStream>,
    heartbeat_timeout_ms: u64,
}

impl UdsPolicySubprocess {
    pub fn spawn(
        workdir: &Path,
        cfg: &PolicyWorkerConfig,
        skill_name: &str,
        arguments: &serde_json::Value,
        heartbeat_timeout_ms: u64,
        startup_timeout_ms: u64,
    ) -> Result<Self, TransportError> {
        let id = SOCKET_SEQ.fetch_add(1, Ordering::Relaxed);
        let socket_path =
            std::env::temp_dir().join(format!("rocode_{}_{}.sock", std::process::id(), id));
        let _ = fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path)
            .map_err(|e| TransportError::Io(format!("bind UDS {}: {e}", socket_path.display())))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| TransportError::Io(e.to_string()))?;

        let mut cmd = Command::new(&cfg.program);
        cmd.current_dir(workdir)
            .args(&cfg.args)
            .arg("--socket")
            .arg(&socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let mut child = cmd.spawn().map_err(|e| {
            let _ = fs::remove_file(&socket_path);
            TransportError::Io(format!("spawn worker '{}': {e}", cfg.program))
        })?;

        let start = Instant::now();
        let startup_budget = Duration::from_millis(startup_timeout_ms.max(200));
        let stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    if let Ok(Some(status)) = child.try_wait() {
                        let _ = fs::remove_file(&socket_path);
                        return Err(TransportError::WorkerCrashed(format!(
                            "exited during startup with {status}"
                        )));
                    }
                    if start.elapsed() > startup_budget {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = fs::remove_file(&socket_path);
                        return Err(TransportError::HeartbeatTimeout(startup_timeout_ms));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = fs::remove_file(&socket_path);
                    return Err(TransportError::Io(format!("accept UDS: {e}")));
                }
            }
        };

        stream
            .set_nonblocking(false)
            .map_err(|e| TransportError::Io(e.to_string()))?;
        let timeout = Duration::from_millis(heartbeat_timeout_ms.max(10));
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| TransportError::Io(e.to_string()))?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(|e| TransportError::Io(e.to_string()))?;

        let reader_stream = stream
            .try_clone()
            .map_err(|e| TransportError::Io(e.to_string()))?;

        let mut proc = Self {
            socket_path,
            child,
            writer: stream,
            reader: BufReader::new(reader_stream),
            heartbeat_timeout_ms,
        };

        let init_resp = proc.exchange(&WorkerRequest::Init {
            skill_name: skill_name.to_string(),
            arguments: arguments.clone(),
        })?;

        match init_resp {
            WorkerResponse::Ready => Ok(proc),
            WorkerResponse::Failed { reason } => Err(TransportError::WorkerCrashed(reason)),
            other => Err(TransportError::Io(format!(
                "unexpected worker handshake response: {other:?}"
            ))),
        }
    }

    pub fn step(&mut self, state: &RobotState) -> Result<WorkerResponse, TransportError> {
        self.exchange(&WorkerRequest::Step {
            state: state.clone(),
        })
    }

    fn exchange(&mut self, req: &WorkerRequest) -> Result<WorkerResponse, TransportError> {
        let mut line =
            serde_json::to_string(req).map_err(|e| TransportError::Io(e.to_string()))?;
        line.push('\n');
        self.writer.write_all(line.as_bytes()).map_err(|e| {
            if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) {
                TransportError::HeartbeatTimeout(self.heartbeat_timeout_ms)
            } else {
                TransportError::WorkerCrashed(e.to_string())
            }
        })?;
        self.writer
            .flush()
            .map_err(|e| TransportError::WorkerCrashed(e.to_string()))?;

        let mut resp_line = String::new();
        let bytes = self.reader.read_line(&mut resp_line).map_err(|e| {
            if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) {
                TransportError::HeartbeatTimeout(self.heartbeat_timeout_ms)
            } else {
                TransportError::WorkerCrashed(e.to_string())
            }
        })?;

        if bytes == 0 {
            return Err(TransportError::WorkerCrashed("EOF from worker".into()));
        }

        serde_json::from_str(resp_line.trim())
            .map_err(|e| TransportError::Io(format!("malformed worker JSON: {e}")))
    }
}

impl Drop for UdsPolicySubprocess {
    fn drop(&mut self) {
        if let Ok(mut bye) = serde_json::to_string(&WorkerRequest::Shutdown) {
            bye.push('\n');
            let _ = self.writer.write_all(bye.as_bytes());
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_file(&self.socket_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_detects_overwritten_frame_slot() {
        let ring = InMemoryTensorRingBuffer::new(4);
        let first = ring.publish_frame("head_rgb", 1_000, &[1, 2, 3]).unwrap();
        assert_eq!(ring.read_frame(&first).unwrap(), vec![1, 2, 3]);

        // Publish 4 more frames to wrap around the 4-slot ring buffer
        for i in 0..4 {
            let _ = ring
                .publish_frame("head_rgb", 2_000 + i, &[9, 9, 9])
                .unwrap();
        }
        assert!(matches!(
            ring.read_frame(&first),
            Err(TransportError::SlotOverwritten(_))
        ));
    }
}

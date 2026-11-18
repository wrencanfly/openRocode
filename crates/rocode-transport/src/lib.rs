//! Pluggable Control Plane & Tensor Plane abstractions.
//!
//! - v0.1: Local IPC control channel + in-memory / UDS frame slots.
//! - v0.3: Zero-copy POSIX Shared Memory Ring Buffer (`ShmRingBuffer` + Seqlock) & Zenoh bridge.

use rocode_proto::FrameRef;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("slot {0} expired or overwritten in ring buffer")]
    SlotOverwritten(u32),
    #[error("transport disconnected: {0}")]
    Disconnected(String),
}

/// Zero-copy-ready abstraction for reading and publishing high-bandwidth sensor/tensor payloads.
pub trait TensorPlane: Send + Sync {
    fn publish_frame(&self, stream_id: &str, timestamp_ns: u64, payload: &[u8]) -> Result<FrameRef, TransportError>;
    fn read_frame(&self, frame: &FrameRef) -> Result<Vec<u8>, TransportError>;
}

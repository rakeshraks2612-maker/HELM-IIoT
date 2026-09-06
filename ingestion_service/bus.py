"""
Asynchronous Streaming Event Backbone.
High-throughput event bus supporting non-blocking stream pub-sub with backpressure and replay buffer.
"""
import asyncio
import structlog
from typing import Callable, List, Dict, Any
from edge_gateway.schemas import CanonicalTelemetryFrame

logger = structlog.get_logger("helm-event-bus")


class TelemetryEventBus:
    """Async streaming event bus with in-memory buffer and pluggable stream connector."""

    def __init__(self, max_buffer_size: int = 5000, persistence_file: str = "telemetry_stream.jsonl"):
        self.max_buffer_size = max_buffer_size
        self.persistence_file = persistence_file
        self._subscribers: List[Callable[[CanonicalTelemetryFrame], Any]] = []
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_buffer_size)
        self._history: List[CanonicalTelemetryFrame] = []
        self._running = False

    def subscribe(self, callback: Callable[[CanonicalTelemetryFrame], Any]):
        """Subscribes an async/sync handler to the live telemetry stream."""
        self._subscribers.append(callback)

    async def publish(self, frame: CanonicalTelemetryFrame):
        """Pushes a canonical frame onto the ingestion queue with backpressure handling and persistence."""
        try:
            self._queue.put_nowait(frame)
            self._history.append(frame)
            if len(self._history) > self.max_buffer_size:
                self._history.pop(0)

            # Append to persistent JSONL stream file
            try:
                with open(self.persistence_file, "a", encoding="utf-8") as f:
                    f.write(frame.model_dump_json() + "\n")
            except Exception:
                pass
        except asyncio.QueueFull:
            logger.warn("event_bus_backpressure_drop", device_id=frame.device_id)

    async def start_consumer_loop(self):
        """Dispatches stream frames to all registered consumers."""
        self._running = True
        while self._running:
            try:
                frame = await self._queue.get()
                for sub in self._subscribers:
                    try:
                        res = sub(frame)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        logger.error("subscriber_error", error=str(e))
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("consumer_loop_error", error=str(e))

    def get_recent_frames(self, count: int = 50) -> List[CanonicalTelemetryFrame]:
        """Returns recent frame history."""
        return self._history[-count:]


# Global event bus singleton
event_bus = TelemetryEventBus()

"""
KeyboardKillswitchController: A class to manage a keyboard-based killswitch for the navigation node.
- Listens for 'k' key to toggle the killswitch state.
- Listens for 'q' key to trigger shutdown and log segment durations.
- Handles SIGINT (Ctrl+C) for graceful shutdown.
"""

import fcntl
import os
import selectors
import signal
import sys
import termios
import threading
from typing import Callable, List

import pandas as pd


class KeyboardKillswitchController:
    def __init__(
        self,
        logger,
        model_name_provider: Callable[[], str],
        segment_durations_provider: Callable[[], List[float]],
        on_killswitch_toggled: Callable[[bool], None],
        on_shutdown: Callable[[], None],
    ):
        self._logger = logger
        self._model_name_provider = model_name_provider
        self._segment_durations_provider = segment_durations_provider
        self._on_killswitch_toggled = on_killswitch_toggled
        self._on_shutdown = on_shutdown

        self.killswitch_active = False
        self.killswitch_state_changed = False
        self._running = False
        self._selector = selectors.DefaultSelector()
        self._old_termios = None

    def start(self):
        self._set_nonblocking(sys.stdin)
        self._selector.register(sys.stdin, selectors.EVENT_READ)
        self._running = True
        threading.Thread(target=self._keyboard_listener, daemon=True).start()
        signal.signal(signal.SIGINT, self._handle_sigint)

    def stop(self):
        self._running = False
        try:
            self._selector.unregister(sys.stdin)
        except Exception:
            pass
        self._selector.close()
        self._restore_terminal()

    def consume_state_change(self) -> bool:
        if not self.killswitch_state_changed:
            return False
        self.killswitch_state_changed = False
        return True

    def _handle_sigint(self, _signum, _frame):
        self._logger.info('Caught SIGINT (Ctrl+C). Cleaning up...')
        self.stop()
        self._on_shutdown()

    def _set_nonblocking(self, file_obj):
        fd = file_obj.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        self._old_termios = termios.tcgetattr(fd)
        new_termios = termios.tcgetattr(fd)
        new_termios[3] = new_termios[3] & ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(fd, termios.TCSANOW, new_termios)

    def _restore_terminal(self):
        if self._old_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._old_termios)

    def _keyboard_listener(self):
        try:
            while self._running:
                events = self._selector.select(timeout=0.1)
                for _, _ in events:
                    input_data = sys.stdin.read(1)
                    if not input_data:
                        continue

                    if input_data == 'k':
                        self.killswitch_active = not self.killswitch_active
                        self.killswitch_state_changed = True
                        self._on_killswitch_toggled(self.killswitch_active)
                        state = 'ACTIVE' if self.killswitch_active else 'INACTIVE'
                        self._logger.info(f'Killswitch toggled: {state}')
                    elif input_data == 'q':
                        self._logger.info('Exiting navigation node...')
                        segment_durations = self._segment_durations_provider()
                        self._logger.info('Segment durations (s):')
                        for i, duration in enumerate(segment_durations):
                            self._logger.info(f'  Segment {i+1}: {duration:.2f} s')

                        df = pd.DataFrame(segment_durations, columns=['duration_s'])
                        model_name = self._model_name_provider()
                        df.to_csv(f'segment_durations_{model_name}.csv', index=False)

                        self.stop()
                        self._on_shutdown()
                        return
        finally:
            self._restore_terminal()

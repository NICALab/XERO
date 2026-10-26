import queue, threading, time, struct
RECORD=struct.Struct("<Q")

class Sink:
    def __init__(self):
        self.q=queue.Queue(2);self.packets=[];self.completed=[];self.error=None
        def worker():
            try:
                while True:
                    item=self.q.get()
                    if item is None:return
                    self.packets.append(item);self.completed.append(time.perf_counter())
            except BaseException as e:self.error=e
        self.thread=threading.Thread(target=worker,daemon=True);self.thread.start()
    def put(self,item):
        if self.error:raise self.error
        self.q.put(item,timeout=60)
    def finish(self):
        self.put(None);self.thread.join(60)
        if self.thread.is_alive():raise RuntimeError('sink stalled')
        if self.error:raise self.error

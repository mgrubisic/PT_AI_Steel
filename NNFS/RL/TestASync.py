import threading
import time
import watchdog
import watchdog.events
import watchdog.observers

global signal


class Handler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['Example.txt'],
                                                             ignore_directories=True, case_sensitive=False)

    def on_modified(self, event):
        print("Modified")
        event_obj.set()


def sjekkSignalet(tekst):
    global signal

    # Start observer
    src_path = "../../../../../../output/"
    event_handler = Handler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=src_path, recursive=False)
    observer.start()

    while not signal:
        time.sleep(0.08)
        continue

    # Signal is changed and code can now proceed.
    # Observer will be disintegrated
    observer.stop()
    print(tekst)


'''if __name__ == '__main__':
    global signal
    signal = False
    print("Signalet: " + str(signal))
    sjekkSignalet("Endret")
'''

def helper_function(event_obj, timeout):
    # Thread has started, but it will wait for 10 seconds for the event
    src_path = "../../../../../../output/"
    event_handler = Handler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=src_path, recursive=False)
    observer.start()
    print("Thread started, for the event to set")

    flag = event_obj.wait()
    if flag:
        print("Event set to true() early")
    else:
        print("Time out occured")
    observer.stop()

if __name__ == '__main__':
    # Initilising an event
    event_obj = threading.Event()

    # Starting the thread who will wait for the event
    thread1 = threading.Thread(target=helper_function, args=(event_obj, 10))
    thread1.start()
    



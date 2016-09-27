from __future__ import division
import sys
import os
import subprocess
import re
import datetime
import collections
import multiprocessing
import tempfile
import signal
import pygraphviz
import pyagg
import numpy as np


FONT = 'lato semibold'

TL_W = 1000
TL_H = 200
TL_MARGIN_X = 10
TL_GUTTER_W = 90
TL_X_SCALE = 9.5
TL_ARROWHEAD_SIZE = 5
TL_ARROW_COLOR = 'green'
TL_XTICK_SIZE = 3
TL_XTICK_OFFS_Y = 8

ANIM_W = 1000
ANIM_H = 750
ANIM_TL_X = 0
ANIM_TL_Y = ANIM_H - TL_H
ANIM_GRAPH_X = 190
ANIM_GRAPH_Y = 100
ANIM_T_SCALE = 1/10
ANIM_GRAPH_PERSISTENCE = 0.1  # seconds
ANIM_ACT_EDGE_COLOR = 'forestgreen'
ANIM_ACT_NODE_COLOR = 'yellow'
ANIM_FPS = 60
ANIM_FRAME_PADDING = 10
ANIM_FRAME_PATH = os.path.join(os.path.dirname(__file__), 'frames')


TimelineArrow = collections.namedtuple('TimelineArrow', 'x y1 y2 angle')
Message = collections.namedtuple('Message', 'src dst t')
Speech = collections.namedtuple('Speech', 'msg t')


def ytick_spacing():
    return TL_H / (len(agent_order) + 1)

def agent_timeline_y(name):
    return (agent_order[name] + 1) * ytick_spacing()

def draw_arrowhead(canvas, x, y, angle, size, **options):
    tf = pyagg.affine.Affine.identity().rotate(angle)
    coords = np.array([[0, 0], [size/2, -size], [-size/2, -size]])
    coords = np.array(tf * coords.T).T + (x, y)
    canvas.draw_polygon(coords.flat, **options)

def draw_arrow(canvas, arrow):
    canvas.draw_line([arrow.x, arrow.y1, arrow.x,arrow.y2],
                     fillcolor=TL_ARROW_COLOR, fillsize=1)
    draw_arrowhead(canvas, arrow.x, arrow.y2, arrow.angle, TL_ARROWHEAD_SIZE,
                   outlinecolor=TL_ARROW_COLOR, fillcolor=TL_ARROW_COLOR)

def draw_ygrid(canvas):
    for a in agent_order.keys():
        y = agent_timeline_y(a)
        canvas.draw_text(a, (TL_MARGIN_X, y), anchor='w',
                         font=FONT, textsize=4)
        canvas.draw_line([TL_GUTTER_W,y, TL_W-TL_MARGIN_X,y],
                         fillcolor=(200,200,200), fillsize=1)

def draw_xgrid(canvas, t0):
    t0 = t0.total_seconds()
    xlim = (TL_W - TL_GUTTER_W - TL_MARGIN_X) / TL_X_SCALE
    ticksize_x = 10 ** np.round(np.log10(xlim)) / 10
    xticks = np.arange(0, t0 + xlim + 1, ticksize_x)
    for time in xticks[xticks >= t0]:
        text = str(datetime.timedelta(0, time))
        x = (time - t0) * TL_X_SCALE + TL_GUTTER_W
        y = TL_H - TL_XTICK_OFFS_Y
        canvas.draw_text(text, (x, y), anchor='s', font=FONT,
                         textsize=TL_XTICK_SIZE)

def merge_agent(name):
    if name in agent_groups:
        name = agent_groups[name]
    return name

line_skip_re = re.compile(r'(<(LOG|EXIT) |$)')
entry_open_re = re.compile(r'<(\w+) T="([^"]+)" ([RS])="([^"]+)">\n$')
entry_close_re = re.compile(r'</\w+>\n$')
body_reject_re = re.compile(r'^\(SORRY ')
sender_re = re.compile(r':sender ([^ )]+)', re.IGNORECASE)
receiver_re = re.compile(r':receiver ([^ )]+)', re.IGNORECASE)
ts_re = re.compile(r'(\d\d):(\d\d):(\d\d(?:\.\d*))')
user_speech_re = re.compile(r':RECEIVER KEYBOARD.*:TEXT "([^<]+)"')
computer_speech_re = re.compile(r'TELL :CONTENT \(SPOKEN :WHAT "([^"]+)"\)')

agent_blacklist = [ 'CHANNELKB', 'DUMMY', 'SPEECH-OUT', 'GRAPHVIZ', 'KEYBOARD',
                    'CONCEPTUALIZER', 'INIT', 'FACILITATOR']

messages = []
speeches = []
tl_arrows = []

t0 = None

agents = ['SPG-AGENT', 'PARSER', 'IM', 'DAGENT', 'CSM', 'MRA', 'DTDA', 'TRA']
agent_order = dict(zip(agents, range(len(agents))))
agent_groups = {
    'TEXTTAGGER': 'PARSER',
    'DEEPSEMLEX': 'PARSER',
    'LEXICONMANAGER': 'PARSER',
}

f = open('facilitator.log')

for line in f:

    if line_skip_re.match(line):
        continue

    event, ts, rel, other = entry_open_re.match(line).groups()
    blines = []
    while not (blines and entry_close_re.match(blines[-1])):
        blines.append(next(f))
    blines.pop()
    body = '\n'.join(blines)

    if event == 'ERROR':
        continue
    assert (event=='R' and rel=='S') or (event=='S' and rel=='R')

    if re.search(body_reject_re, body):
        continue

    ts_hour, ts_min, ts_sec = re.findall(ts_re, ts)[0]
    total_sec = ((int(ts_hour) * 24) + int(ts_min)) * 60 + float(ts_sec)
    t = datetime.timedelta(0, total_sec)

    if other == 'KEYBOARD':
        text = None
        if user_speech_re.search(body):
            text = 'User: ' + user_speech_re.findall(body)[0]
        elif computer_speech_re.search(body):
            text = 'Computer: ' + computer_speech_re.findall(body)[0]
        if text is not None:
            # Rudimentary HTML tag stripping.
            text = re.sub(r'</?\w+>', '', text)
            speeches.append(Speech(text, t.total_seconds()))
            continue

    s_match = re.search(sender_re, body)
    r_match = re.search(receiver_re, body)
    sender = receiver = None
    if s_match:
        sender = s_match.groups()[0].upper()
        if rel == 'S':
            assert sender == other
    else:
        if rel == 'S':
            sender = other
    if r_match:
        receiver = r_match.groups()[0].upper()
        if rel == 'R':
            assert receiver == other
    else:
        if rel == 'R':
            receiver = other
    if (sender in agent_blacklist or receiver in agent_blacklist
        or sender == receiver):
        continue

    if not (sender and receiver):
        #print line + body
        continue
    sender = merge_agent(sender)
    receiver = merge_agent(receiver)

    if t0 is None:
        t0 = t

    # Add message (graph edge).
    messages.append(Message(sender, receiver, t.total_seconds()))

    # Add timeline arrow.
    x = TL_GUTTER_W + (t.total_seconds()-t0.total_seconds()) * TL_X_SCALE
    y1 = agent_timeline_y(sender)
    y2 = agent_timeline_y(receiver)
    angle = 0 if y1 < y2 else 180
    tl_arrows.append(TimelineArrow(x, y1, y2, angle))

tfinal = t


g = pygraphviz.AGraph(directed=True)
g.node_attr['fontname'] = FONT
g.node_attr['fontsize'] = 11
for m in messages:
    g.add_edge(m.src, m.dst)
g.layout(prog='dot')


anim_seconds = (tfinal - t0).total_seconds() * ANIM_T_SCALE
num_frames = int(np.ceil(anim_seconds * ANIM_FPS)) + ANIM_FRAME_PADDING

base_frame = pyagg.Canvas(TL_W, TL_H, background='white')
base_frame.default_unit = 'px'
draw_ygrid(base_frame)
draw_xgrid(base_frame, t0)
for a in tl_arrows:
    draw_arrow(base_frame, a)
base_frame.drawer.flush()

if not os.path.exists(ANIM_FRAME_PATH):
    os.mkdir(ANIM_FRAME_PATH)

worker_pids = {}

def kill_workers():
    for pid in worker_pids:
        os.kill(pid, signal.SIGINT)

def cleanup_workers():
    while worker_pids:
        pid, _ = os.wait()
        del worker_pids[pid]

def sigint_handler(sig, frame):
    kill_workers()
    cleanup_workers()
    sys.exit()

num_workers = multiprocessing.cpu_count()
signal.signal(signal.SIGINT, signal.SIG_IGN)
for worker_num in range(num_workers):
    pid = os.fork()
    if pid:
        # Master will collect the child pid and continue looping.
        worker_pids[pid] = worker_num
    else:
        # Worker will proceed from here with worker_num set appropriately.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        process_type = 'worker'
        break
else:
    process_type = 'master'
    signal.signal(signal.SIGINT, sigint_handler)

if process_type == 'worker':

    ANIM_GRAPH_TMP_FILE = tempfile.NamedTemporaryFile(suffix='.png')

    # Split work equally between workers.
    for f in xrange(worker_num, num_frames, num_workers):

        print "%d: %d/%d" % (worker_num, f, num_frames)
        sys.stdout.flush()

        t_rel = f / ANIM_FPS / ANIM_T_SCALE
        t_abs = t_rel + t0.total_seconds()

        # Draw graph.
        g2 = g.copy()
        g2.has_layout = True
        for m in messages:
            if m.t <= t_abs <= m.t + ANIM_GRAPH_PERSISTENCE / ANIM_T_SCALE:
                g2.get_edge(m.src, m.dst).attr['color'] = ANIM_ACT_EDGE_COLOR
                for n in m.src, m.dst:
                    node = g2.get_node(n)
                    node.attr['style'] = 'filled'
                    node.attr['fillcolor'] = ANIM_ACT_NODE_COLOR

        g2.draw(ANIM_GRAPH_TMP_FILE.name)
        graph_frame = pyagg.load(ANIM_GRAPH_TMP_FILE.name)
        graph_frame.crop(0, 0, graph_frame.width-1, graph_frame.height-1)

        # Draw timeline.
        tl_frame = base_frame.copy()
        x = t_rel * TL_X_SCALE + TL_GUTTER_W
        margin = ytick_spacing()
        y1 = margin
        y2 = TL_H - margin
        tl_frame.draw_line([x,y1, x,y2], fillcolor='black', fillsize=3)
        tl_frame.drawer.flush()

        frame = pyagg.Canvas(ANIM_W, ANIM_H, background='white')
        frame.paste(graph_frame, (ANIM_GRAPH_X, ANIM_GRAPH_Y))
        frame.paste(tl_frame, (ANIM_TL_X, ANIM_TL_Y))

        for s in reversed(speeches):
            if s.t <= t_abs:
                frame.draw_text(s.msg, (ANIM_W / 2, 10), anchor='n',
                                font=FONT, textsize=6)
                break

        frame.save(os.path.join(ANIM_FRAME_PATH, '%05d.png' % f))

    # Workers just exist after they finish their work.
    sys.exit()

else:

    # Master waits for workers then restores SIGINT handler.
    cleanup_workers()
    signal.signal(signal.SIGINT, signal.SIG_DFL)

print

cmd = ('ffmpeg -i %s/%%05d.png -y -vcodec libx264 -profile:v main -level 3'
       ' -pix_fmt yuv420p -crf 21 -an %s' % (ANIM_FRAME_PATH, 'dialog.mp4'))
print cmd
print "\n\n\n\n"
subprocess.call(cmd.split(' '))

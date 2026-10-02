"""Web regressions and live client: python tests/test_ws_flow.py --url ws://localhost:8000/ws."""
import argparse
import base64
import io
import json
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.server import Metrics, WebSink, create_app
from bench.make_synthetic_boards import render_board
from core.config import ROOT
from speech.scheduler import AudioItem, Priority
from speech.stt import Transcript
from transit.bus_detector import PassthroughBackend
from transit.ocr import OCRLine


def board_jpeg():
    frame,_ = render_board("21G")
    ok,data = cv2.imencode('.jpg',frame)
    assert ok
    return data.tobytes()


class FakeTTS:
    def synthesize(self,text):
        return np.zeros(800,dtype=np.int16),16000
class FakeOCR:
    def read(self,image):
        return [OCRLine('21 G',.99,((0,0),(100,0),(100,40),(0,40)))]
class FakeSTT:
    def start(self,sr=16000):
        self.audio = b''
    def feed(self,chunk):
        self.audio += chunk
        return 'find bus twenty one g'
    def finish(self,endpoint=None):
        return Transcript('find bus twenty one g',.95,1.5)


def fake_bundle():
    return SimpleNamespace(metrics=Metrics(),load_times={'stt':0},routes={'21G','23C'},
                           route_info={'21G':{'route':'21G','variants':['21G','21G CT'],'long_names':['Kilambakkam TO Besant Nagar']}},
                           tts=FakeTTS(),ocr=FakeOCR(),detector=SimpleNamespace(backend=PassthroughBackend(),detect=PassthroughBackend().detect),recognizer=FakeSTT)


def receive_until(ws,predicate,limit=100):
    for _ in range(limit):
        message = ws.receive_json()
        if predicate(message):
            return message
    pytest.fail('Expected WebSocket message was not received')


@pytest.mark.parametrize('base64_frame',[False,True])
def test_ws_text_frame_and_bus_speech(base64_frame):
    from fastapi.testclient import TestClient
    with TestClient(create_app(fake_bundle)) as client:
        with client.websocket_connect('/ws') as ws:
            ws.send_json({'type':'text_command','text':'find bus 21G'})
            intent = receive_until(ws,lambda m:m['type']=='intent')
            assert intent['intent']=='FIND_BUS' and intent['target']=='21G'
            assert intent['state']['target_bus']['long_names']==['Kilambakkam TO Besant Nagar']
            state = receive_until(ws,lambda m:m['type']=='state' and m['target_bus'])
            assert state['target_bus']['long_names']==['Kilambakkam TO Besant Nagar']
            ws.send_json({'type':'source_mode','mode':'board'})
            payload={'type':'frame','id':7}
            if base64_frame:payload['jpeg_b64']=base64.b64encode(board_jpeg()).decode()
            ws.send_json(payload)
            if not base64_frame:ws.send_bytes(board_jpeg())
            seen_overlay = seen_speech = False
            for _ in range(100):
                message = ws.receive_json()
                if message['type']=='overlay':
                    assert message['matched_route']=='21G'
                    assert message['status']=='confirmed'
                    assert message['boxes'][0]['ocr_attempts'][0]['crop']=='full'
                    assert message['boxes'][0]['ocr_lines'][0]['box_frame']
                    seen_overlay = True
                if message['type']=='speak':
                    ws.send_json({'type':'playback_done','id':message['id']})
                    if message['priority']==2:
                        assert 'Bus 21G detected' in message['text']
                        assert base64.b64decode(message['audio_b64_wav']).startswith(b'RIFF')
                        seen_speech = True
                if seen_overlay and seen_speech:
                    break
            else:pytest.fail('No bus speech received')


def test_audio_stream_risk_and_error_recovery():
    from fastapi.testclient import TestClient
    with TestClient(create_app(fake_bundle)) as client:
        with client.websocket_connect('/ws') as ws:
            ws.send_bytes(b'\0\0')
            assert receive_until(ws,lambda m:m['type']=='error')
            ws.send_json({'type':'audio_start'})
            ws.send_bytes(bytes(1600))
            partial = receive_until(ws,lambda m:m['type']=='partial_transcript' and m['text'])
            assert 'twenty one' in partial['text']
            ws.send_json({'type':'audio_end'})
            transcript = receive_until(ws,lambda m:m['type']=='transcript')
            assert transcript['latency_ms']==1.5
            assert receive_until(ws,lambda m:m['type']=='intent')['target']=='21G'
            ws.send_json({'type':'mock_risk','risk':'CRITICAL','object':'motorcycle','direction':'left'})
            critical = receive_until(ws,lambda m:m['type']=='speak' and m['priority']==0)
            assert critical['text']=='Motorcycle left!'
            ws.send_json({'type':'playback_done','id':critical['id']})


def test_http_image_routes_metrics_and_static():
    from fastapi.testclient import TestClient
    with TestClient(create_app(fake_bundle)) as client:
        assert 'Voice + Transit' in client.get('/').text
        assert client.get('/static/pcm-worklet.js').status_code==200
        assert client.get('/static/js/main.js').headers['cache-control']=='no-cache'
        assert client.get('/static/samples/index.json').json()
        assert client.get('/api/routes?q=21').json()['routes']==['21G']
        assert client.get('/api/route/21g').json()=={'route':'21G','variants':['21G','21G CT'],'long_names':['Kilambakkam TO Besant Nagar']}
        assert client.get('/api/route/23C').json()['long_names']==[]
        assert client.get('/api/route/999').status_code==404
        health=client.get('/api/health').json()
        assert health['routes']==2 and health['load_time_ms']=={'stt':0} and 'commit' in health['git']
        response=client.post('/api/image',files={'file':('board.jpg',board_jpeg(),'image/jpeg')},data={'command':'find bus 21G','board':'true'})
        assert response.status_code==200
        result=response.json()
        assert result['overlay']['matched_route']=='21G'
        assert base64.b64decode(result['annotated_jpeg_b64']).startswith(b'\xff\xd8')
        assert 'p95' in client.get('/api/metrics').json()['latency_ms']['ocr']
        assert client.post('/api/image',files={'file':('bad.jpg',b'bad','image/jpeg')}).status_code==400


def test_web_sink_interrupt_completion_and_duration_timeout():
    messages=[];now=[0.0]
    sink=WebSink(messages.append,clock=lambda:now[0])
    sink.prepare_item(AudioItem('Bus 21G detected.',Priority.BUS,'bus:21G'))
    sink.play(np.zeros(16000,dtype=np.int16),16000,'bus')
    with wave.open(io.BytesIO(base64.b64decode(messages[-1]['audio_b64_wav']))) as wav:
        assert wav.getframerate()==16000 and wav.getnframes()==16000
    sink.playback_done('stale')
    assert sink.is_playing
    sink.stop()
    assert messages[-1]['type']=='stop_audio' and not sink.is_playing
    sink.prepare_item(AudioItem('Vehicle approaching right.',Priority.HIGH,'risk:car:right'))
    sink.play(np.zeros(16000,dtype=np.int16),16000,'risk')
    assert messages[-1]['interrupt'] and messages[-1]['priority']==1
    sink.playback_done('bus');assert sink.is_playing
    now[0]=1.76;assert not sink.is_playing
    sink.play(np.zeros(10,dtype=np.int16),16000,'next')
    sink.playback_done('next');assert not sink.is_playing


def test_recognizer_sessions_share_model_without_decoder_state():
    from speech.stt import VoskSTT
    parent=object.__new__(VoskSTT)
    parent.config=object();parent.model=object();parent.grammar=['find','[unk]'];parent.recognizer=object();parent._segments=[{}]
    child=parent.new_session()
    assert child.model is parent.model and child.recognizer is None and not child._segments
    child.grammar.append('bus');assert 'bus' not in parent.grammar


def run_live(url):
    from websockets.sync.client import connect
    with connect(url,max_size=8*1024*1024,open_timeout=10) as ws:
        ws.send(json.dumps({'type':'source_mode','mode':'board'}))
        ws.send(json.dumps({'type':'text_command','text':'find bus 21G'}))
        ws.send(json.dumps({'type':'frame','id':1}))
        ws.send(board_jpeg())
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            message=json.loads(ws.recv(timeout=max(.1,deadline-time.monotonic())))
            if message['type']=='intent':print('INTENT',message['intent'],message['target'])
            if message['type']=='overlay':print('OVERLAY route=',message['matched_route'],'OCR=',message['ocr_text'])
            if message['type']=='speak':
                print('SPEAK',message['priority'],message['text'],'WAV bytes=',len(base64.b64decode(message['audio_b64_wav'])))
                ws.send(json.dumps({'type':'playback_done','id':message['id']}))
                if message['priority']==2 and '21G' in message['text']:
                    assert message['audio_b64_wav']
                    print('PASS: typed command -> synthetic board -> BUS-priority WAV containing 21G')
                    return
        raise AssertionError('No matching bus speech received')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='ws://localhost:8000/ws')
    run_live(parser.parse_args().url)


def test_worklet_resampling_counts_amplitude_and_flush():
    import shutil
    import subprocess
    if not shutil.which('node'):
        pytest.skip('Node absent; only needed to execute AudioWorklet unit harness')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('app/static/pcm-worklet.js','utf8');
for(const sr of [44100,48000]) {
  let Processor;const packets=[];
  const sandbox={sampleRate:sr,AudioWorkletProcessor:class {constructor(){this.port={postMessage:p=>packets.push(p)}}},
    registerProcessor:(name,klass)=>{Processor=klass;},Int16Array,Math};
  vm.runInNewContext(source,sandbox);const processor=new Processor();
  processor.port.onmessage({data:'start'});
  for(let offset=0;offset<sr;offset+=128){const size=Math.min(128,sr-offset);processor.process([[new Float32Array(size).fill(.5)]]);}
  processor.port.onmessage({data:'stop'});
  const samples=packets.filter(p=>p.type==='pcm').flatMap(p=>Array.from(new Int16Array(p.buffer)));
  assert.strictEqual(samples.length,16000);assert(samples.every(v=>Math.abs(v-16384)<=1));
  assert.strictEqual(packets.at(-1).type,'stopped');
  const before=packets.length;processor.process([[new Float32Array(128).fill(1)]]);assert.strictEqual(packets.length,before);
}
'''
    subprocess.run(['node','-e',script],cwd=ROOT,check=True,capture_output=True,text=True)


def test_web_scheduler_preempts_bus_with_stop_audio_before_warning():
    from fastapi.testclient import TestClient
    class LongTTS:
        def synthesize(self,text):
            return np.zeros(8000,dtype=np.int16),16000
    def bundle():
        result=fake_bundle();result.tts=LongTTS();return result
    with TestClient(create_app(bundle)) as client:
        with client.websocket_connect('/ws') as ws:
            ws.send_json({'type':'text_command','text':'find bus 21G'})
            acknowledgement=receive_until(ws,lambda m:m['type']=='speak')
            ws.send_json({'type':'playback_done','id':acknowledgement['id']})
            ws.send_json({'type':'frame','id':1})
            ws.send_bytes(board_jpeg())
            bus=receive_until(ws,lambda m:m['type']=='speak' and m['priority']==2)
            ws.send_json({'type':'mock_risk','risk':'HIGH','object':'car','direction':'right'})
            receive_until(ws,lambda m:m['type']=='stop_audio')
            warning=receive_until(ws,lambda m:m['type']=='speak' and m['priority']==1)
            assert warning['interrupt'] and warning['text']=='Vehicle approaching right.'
            # Late completion of the interrupted bus cannot complete the warning.
            ws.send_json({'type':'playback_done','id':bus['id']})
            ws.send_json({'type':'playback_done','id':warning['id']})

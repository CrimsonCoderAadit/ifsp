/* Weighted averaging resampler: native rate -> 16 kHz, preserving fractional phase. */
class PCM16Processor extends AudioWorkletProcessor {
  constructor() {
    super(); this.running = false; this.ratio = sampleRate / 16000;
    this.sum = 0; this.weight = 0; this.pending = [];
    this.port.onmessage = ({data}) => {
      if (data === 'start') { this.running = true; this.sum = this.weight = 0; this.pending = []; }
      if (data === 'stop') { this.running = false; this.flush(); this.port.postMessage({type:'stopped'}); }
    };
  }
  flush() {
    if (!this.pending.length) return;
    const packet = new Int16Array(this.pending); this.pending = [];
    this.port.postMessage({type:'pcm',buffer:packet.buffer},[packet.buffer]);
  }
  process(inputs) {
    const channels = inputs[0];
    if (!this.running || !channels || !channels.length) return true;
    for (let i=0;i<channels[0].length;i++) {
      let value=0; for (const channel of channels) value += channel[i]/channels.length;
      let remainder=1;
      while (remainder>1e-9) {
        const part=Math.min(remainder,this.ratio-this.weight);
        this.sum += value*part; this.weight += part; remainder -= part;
        if (this.weight >= this.ratio-1e-9) {
          const sample=Math.max(-1,Math.min(1,this.sum/this.ratio));
          this.pending.push(Math.round(sample*(sample<0?32768:32767)));
          this.sum=this.weight=0;
          if (this.pending.length>=800) this.flush();
        }
      }
    }
    return true; // Output remains silent; mic audio is never monitored.
  }
}
registerProcessor('pcm16',PCM16Processor);

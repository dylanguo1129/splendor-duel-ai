// Client for train/infer_server.py (one per worker thread, one outstanding request at a time).
import { connect, Socket } from "node:net";
import { NUM_ACTIONS } from "../engine/constants";
import { NUM_FEATURES } from "../engine/features";
import { toHalf } from "./records";

export const OUT_PER_EVAL = 2 + NUM_ACTIONS; // value, logits, moves left / MOVES_SCALE
export const MOVES_SCALE = 100; // train/train.py MOVES_SCALE

const HALF_TABLE = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();
export const fromHalf = (h: number) => HALF_TABLE[h];

function writeHalves(msg: Buffer, at: number, feats: Float32Array, count: number): void {
  if ((msg.byteOffset + at) % 2 !== 0) throw new Error("unaligned buffer");
  const u16 = new Uint16Array(msg.buffer, msg.byteOffset + at, count);
  for (let i = 0; i < count; i++) {
    const v = feats[i];
    u16[i] = v === 0 ? 0 : toHalf(v);
  }
}

export class GpuClient {
  private chunks: Buffer[] = [];
  private have = 0;
  private waiter: { need: number; resolve: (b: Buffer) => void; reject: (e: Error) => void } | null = null;
  private closed: Error | null = null;

  private constructor(private sock: Socket) {
    sock.setNoDelay(true);
    sock.on("data", (d: Buffer) => {
      this.chunks.push(d);
      this.have += d.length;
      this.pump();
    });
    const fail = (e: Error) => {
      this.closed = e;
      this.waiter?.reject(e);
      this.waiter = null;
    };
    sock.on("error", fail);
    sock.on("close", () => fail(new Error("inference server connection closed")));
  }

  static connect(port: number): Promise<GpuClient> {
    return new Promise((resolve, reject) => {
      const s = connect({ host: "127.0.0.1", port }, () => resolve(new GpuClient(s)));
      s.once("error", reject);
    });
  }

  private pump(): void {
    if (!this.waiter || this.have < this.waiter.need) return;
    const all = this.chunks.length === 1 ? this.chunks[0] : Buffer.concat(this.chunks);
    const { need, resolve } = this.waiter;
    this.waiter = null;
    const rest = all.subarray(need);
    this.chunks = rest.length ? [rest] : [];
    this.have = rest.length;
    resolve(all.subarray(0, need));
  }

  private read(need: number): Promise<Buffer> {
    if (this.closed) return Promise.reject(this.closed);
    return new Promise((resolve, reject) => {
      this.waiter = { need, resolve, reject };
      this.pump();
    });
  }

  async load(slot: number, path: string): Promise<void> {
    const p = Buffer.from(path, "utf8");
    const head = Buffer.alloc(4);
    head.write("L", 0);
    head.writeUInt8(slot, 1);
    head.writeUInt16LE(p.length, 2);
    this.sock.write(Buffer.concat([head, p]));
    const r = await this.read(1);
    if (r[0] === 0x4b) return; // 'K'
    const len = (await this.read(2)).readUInt16LE(0);
    throw new Error(`model load failed: ${(await this.read(len)).toString()}`);
  }

  /** feats: n*F float32 features. Returns n*OUT_PER_EVAL float32 (value, logits, moves left). */
  async eval(slot: number, feats: Float32Array, n: number): Promise<Float32Array> {
    const msg = Buffer.alloc(6 + n * NUM_FEATURES * 2);
    msg.write("V", 0);
    msg.writeUInt8(slot, 1);
    msg.writeUInt32LE(n, 2);
    writeHalves(msg, 6, feats, n * NUM_FEATURES);
    this.sock.write(msg);
    return (await this.readResult(n))[0];
  }

  /** Several slots in one request (one round trip); results in the same order. */
  async evalMulti(parts: { slot: number; feats: Float32Array; n: number }[]): Promise<Float32Array[]> {
    let size = 2;
    for (const p of parts) size += 6 + p.n * NUM_FEATURES * 2;
    const msg = Buffer.alloc(size);
    msg.write("M", 0);
    msg.writeUInt8(parts.length, 1);
    let o = 2;
    for (const p of parts) {
      msg.writeUInt8(p.slot, o);
      msg.writeUInt32LE(p.n, o + 1);
      o += 6; // slot, n, 1 pad byte keeps the float16 data aligned
      writeHalves(msg, o, p.feats, p.n * NUM_FEATURES);
      o += p.n * NUM_FEATURES * 2;
    }
    this.sock.write(msg);
    return this.readResult(...parts.map((p) => p.n));
  }

  private async readResult(...ns: number[]): Promise<Float32Array[]> {
    const total = ns.reduce((a, b) => a + b, 0);
    const head = await this.read(5);
    if (head[0] !== 0x52 || head.readUInt32LE(1) !== total) throw new Error("bad eval response");
    const body = await this.read(total * OUT_PER_EVAL * 2);
    let o = 0;
    return ns.map((n) => {
      const out = new Float32Array(n * OUT_PER_EVAL);
      for (let i = 0; i < out.length; i++, o += 2) out[i] = HALF_TABLE[body.readUInt16LE(o)];
      return out;
    });
  }

  close(): void {
    this.sock.destroy();
  }
}

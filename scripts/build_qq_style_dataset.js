const fs = require("fs");
const path = require("path");

const SOURCE = process.argv[2] || "E:/TencentFile/QQrecv/dialogs.jsonl";
const OUTPUT = process.argv[3] || "E:/TencentFile/QQrecv/qq_style_2024plus";
const CUTOFF = 1704038400; // 2024-01-01 00:00:00 Asia/Shanghai
const MAX_CHARS = 200;
const TARGET_QQ = "3328717966";

const SYSTEM_PROMPT =
  "根据群聊上下文，用目标人物一贯的口吻生成一条短回复。保持原始用词、标点、语气词、错别字和表情习惯，不要解释，不要补充背景，不要写成长文。";

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function writeJsonl(file, rows) {
  fs.writeFileSync(
    file,
    rows.map((row) => JSON.stringify(row, null, 0)).join("\n") + (rows.length ? "\n" : ""),
    "utf8",
  );
}

function textLength(value) {
  return [...String(value)].length;
}

function quantile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q))];
}

function increment(map, key, amount = 1) {
  map.set(key, (map.get(key) || 0) + amount);
}

function topEntries(map, limit = 50) {
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([value, count]) => ({ value, count }));
}

function mediaLabel(type) {
  switch (type) {
    case "image":
      return "[图片]";
    case "qqface":
      return "[QQ表情]";
    case "marketface":
      return "[商城表情]";
    case "ptt":
      return "[语音]";
    case "file":
      return "[文件]";
    case "video":
      return "[视频]";
    case "reply":
      return "[引用]";
    case "ark":
      return "[卡片]";
    case "gray":
      return "[灰条]";
    default:
      return "[其他媒体]";
  }
}

function sanitizeText(value) {
  return String(value)
    .replace(/https?:\/\/\S+/gi, "[链接]")
    .replace(/[A-Za-z]:\\(?:[^\\\n ]+\\)*[^\\\n ]+/g, "[本地路径]");
}

function containsUnsafeReference(value) {
  return /https?:\/\/|[A-Za-z]:\\(?:[^\\\n ]+\\)*[^\\\n ]+/i.test(String(value));
}

function displayMessage(message, target = false) {
  const name = target
    ? "他"
    : message.sender_name || String(message.sender_qq || "未知用户");
  const text = message.text == null ? "" : sanitizeText(message.text).trim();
  const labels = [];
  for (const media of message.media || []) {
    const label = mediaLabel(media.type);
    if (!labels.includes(label)) labels.push(label);
  }
  const body = [text, ...labels].filter(Boolean).join(" ");
  return `${name}: ${body || "[空消息]"}`;
}

function buildExample(raw) {
  const context = (raw.context || [])
    .filter((message) => Number(message.ts) >= CUTOFF)
    .map((message) => displayMessage(message, String(message.sender_qq) === TARGET_QQ))
    .join("\n");

  const output = String(raw.reply.text).trim();
  const userContent = context || "(上下文为空)";
  return {
    id: String(raw.reply.id),
    group_qq: raw.group_qq,
    time: raw.reply.time,
    ts: Number(raw.reply.ts),
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userContent },
      { role: "assistant", content: output },
    ],
  };
}

function toAlpaca(example) {
  return {
    instruction: SYSTEM_PROMPT,
    input: example.messages[1].content,
    output: example.messages[2].content,
    group_qq: example.group_qq,
    time: example.time,
    ts: example.ts,
  };
}

function makeProfile(examples, rawStats) {
  const lengths = examples.map((item) => textLength(item.messages[2].content));
  const punctuation = new Map();
  const emoji = new Map();
  const ngrams = new Map();
  const exactOutputs = new Map();

  for (const item of examples) {
    const output = item.messages[2].content;
    increment(exactOutputs, output);
    for (const char of output) {
      if ("，。！？：；、,.!?~～…".includes(char)) increment(punctuation, char);
    }
    for (const symbol of output.match(/\p{Extended_Pictographic}/gu) || []) {
      increment(emoji, symbol);
    }
    const chars = [...output];
    for (let size = 2; size <= 4; size += 1) {
      for (let i = 0; i + size <= chars.length; i += 1) {
        const gram = chars.slice(i, i + size).join("");
        if (!/[\u4e00-\u9fff]/.test(gram)) continue;
        increment(ngrams, gram);
      }
    }
  }

  return {
    cutoff: "2024-01-01T00:00:00+08:00",
    target_qq: TARGET_QQ,
    source: path.basename(SOURCE),
    sample_count: examples.length,
    date_range: {
      first: examples[0]?.time || null,
      last: examples.at(-1)?.time || null,
    },
    length_chars: {
      median: quantile(lengths, 0.5),
      p90: quantile(lengths, 0.9),
      max: Math.max(...lengths, 0),
    },
    top_output_ngrams: topEntries(ngrams, 100),
    punctuation_counts: topEntries(punctuation, 30),
    emoji_counts: topEntries(emoji, 30),
    repeated_outputs: topEntries(exactOutputs, 30).filter((item) => item.count > 1),
    raw_filter_stats: rawStats,
  };
}

function main() {
  ensureDir(OUTPUT);
  const lines = fs.readFileSync(SOURCE, "utf8").split(/\r?\n/).filter(Boolean);
  const examples = [];
  const rawStats = {
    source_lines: lines.length,
    after_cutoff: 0,
    before_cutoff: 0,
    target_text: 0,
    target_nontext: 0,
    target_text_with_link: 0,
    target_text_too_long: 0,
    target_text_with_media: 0,
    context_messages_removed_by_cutoff: 0,
    context_messages_retained: 0,
  };
  const groups = new Map();
  const contextSenders = new Map();
  const replyMedia = new Map();
  const allOutputLengths = [];

  for (const line of lines) {
    const raw = JSON.parse(line);
    const reply = raw.reply || {};
    const ts = Number(reply.ts);

    if (ts < CUTOFF) {
      rawStats.before_cutoff += 1;
      continue;
    }
    rawStats.after_cutoff += 1;
    increment(groups, String(raw.group_qq));

    if (reply.text == null) {
      rawStats.target_nontext += 1;
      for (const media of reply.media || []) increment(replyMedia, media.type || "?");
      continue;
    }
    rawStats.target_text += 1;
    if (containsUnsafeReference(reply.text)) {
      rawStats.target_text_with_link += 1;
      continue;
    }
    if (textLength(reply.text) >= MAX_CHARS) {
      rawStats.target_text_too_long += 1;
      continue;
    }
    if ((reply.media || []).length) rawStats.target_text_with_media += 1;

    const context = raw.context || [];
    rawStats.context_messages_removed_by_cutoff += context.filter(
      (message) => Number(message.ts) < CUTOFF,
    ).length;
    rawStats.context_messages_retained += context.filter(
      (message) => Number(message.ts) >= CUTOFF,
    ).length;
    for (const message of context) {
      if (Number(message.ts) >= CUTOFF) increment(contextSenders, String(message.sender_qq));
    }

    const example = buildExample(raw);
    examples.push(example);
    allOutputLengths.push(textLength(example.messages[2].content));
  }

  examples.sort((a, b) => a.ts - b.ts || a.id.localeCompare(b.id));
  const trainEnd = Math.floor(examples.length * 0.8);
  const valEnd = Math.floor(examples.length * 0.9);
  const train = examples.slice(0, trainEnd);
  const val = examples.slice(trainEnd, valEnd);
  const test = examples.slice(valEnd);
  const testAnswers = test.map((example) => ({
    id: example.id,
    group_qq: example.group_qq,
    time: example.time,
    ts: example.ts,
    output: example.messages[2].content,
  }));

  writeJsonl(path.join(OUTPUT, "style_train.jsonl"), train);
  writeJsonl(path.join(OUTPUT, "style_val.jsonl"), val);
  writeJsonl(
    path.join(OUTPUT, "style_test.jsonl"),
    test.map((example) => ({
      id: example.id,
      group_qq: example.group_qq,
      time: example.time,
      ts: example.ts,
      messages: example.messages.slice(0, 2),
    })),
  );
  writeJsonl(path.join(OUTPUT, "style_test_answers.jsonl"), testAnswers);
  writeJsonl(path.join(OUTPUT, "style_train_alpaca.jsonl"), train.map(toAlpaca));
  writeJsonl(path.join(OUTPUT, "style_val_alpaca.jsonl"), val.map(toAlpaca));

  const profile = makeProfile(examples, {
    ...rawStats,
    groups: topEntries(groups, 20),
    context_senders: topEntries(contextSenders, 50),
    reply_media_after_cutoff: topEntries(replyMedia, 30),
    output_length_chars: {
      median: quantile(allOutputLengths, 0.5),
      p90: quantile(allOutputLengths, 0.9),
      max: Math.max(...allOutputLengths, 0),
    },
  });
  fs.writeFileSync(
    path.join(OUTPUT, "persona_profile.json"),
    JSON.stringify(profile, null, 2) + "\n",
    "utf8",
  );

  const stats = {
    cutoff: profile.cutoff,
    source: SOURCE,
    output: OUTPUT,
    counts: {
      train: train.length,
      val: val.length,
      test: test.length,
      test_answers: testAnswers.length,
    },
    split_boundaries: {
      train_last: train.at(-1)?.time || null,
      val_first: val[0]?.time || null,
      val_last: val.at(-1)?.time || null,
      test_first: test[0]?.time || null,
    },
    filters: rawStats,
    groups: topEntries(groups, 20),
    context_senders: topEntries(contextSenders, 50),
    output_length_chars: {
      median: quantile(allOutputLengths, 0.5),
      p90: quantile(allOutputLengths, 0.9),
      max: Math.max(...allOutputLengths, 0),
    },
  };
  fs.writeFileSync(path.join(OUTPUT, "stats.json"), JSON.stringify(stats, null, 2) + "\n", "utf8");
  console.log(JSON.stringify(stats, null, 2));
}

main();

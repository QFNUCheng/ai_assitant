/* 儒家经典知识库 · 前端交互
 *
 * 纯 Vue 3（无构建步骤），直接调后端 API：
 *   GET    /health
 *   GET    /kb                          知识库列表
 *   POST   /kb/upload                   上传（需 X-Upload-Token）
 *   GET    /kb/tasks/{id}               任务进度
 *   GET    /kb/{id}/files               文件列表
 *   DELETE /kb/{id}/files/{file_id}     删文件
 *   DELETE /kb/{id}                     删库
 *   POST   /sessions  GET /sessions/{id}/messages  POST /chat
 */

const { createApp, ref, reactive, computed, onMounted, nextTick } = Vue;

// 同源部署时留空即可；本地开发可改成 http://127.0.0.1:8000
const API = "";

async function api(path, { method = "GET", body, token, raw } = {}) {
  const headers = {};
  if (token) headers["X-Upload-Token"] = token;
  if (body && !raw) headers["Content-Type"] = "application/json";
  const res = await fetch(API + path, {
    method,
    headers,
    body: raw ? body : body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (!res.ok) {
    const err = new Error(data.detail || `HTTP ${res.status}`);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

createApp({
  setup() {
    // ---- 状态 ----
    const healthy = ref(null);
    const token = ref(localStorage.getItem("kb_token") || "");
    const kbs = ref([]);
    const activeKb = ref("rujia");
    const files = ref([]);
    const sessions = ref([]);
    const sessionId = ref(null);
    const messages = ref([]);
    const question = ref("");
    const busy = ref(false);
    const uploading = ref(false);
    const dragOver = ref(false);
    const toast = reactive({ text: "", kind: "" });
    const task = ref(null);
    const chatBox = ref(null);

    const activeKbName = computed(() => {
      const found = kbs.value.find((k) => k.id === activeKb.value);
      return found ? found.name : activeKb.value;
    });

    function say(text, kind = "ok") {
      toast.text = text;
      toast.kind = kind;
      setTimeout(() => { if (toast.text === text) toast.text = ""; }, 6000);
    }

    function saveToken() {
      localStorage.setItem("kb_token", token.value);
      say("令牌已保存在本机浏览器");
    }

    // ---- 知识库 ----
    async function loadKbs() {
      try {
        const d = await api("/kb");
        kbs.value = d.knowledge_bases || [];
      } catch (e) { say("读取知识库失败：" + e.message, "err"); }
    }

    async function loadFiles(kbId) {
      activeKb.value = kbId;
      files.value = [];
      if (kbId === "rujia") return; // 内置库不支持文件管理
      try {
        const d = await api(`/kb/${kbId}/files`, { token: token.value });
        files.value = d.files || [];
      } catch (e) { say("读取文件失败：" + e.message, "err"); }
    }

    // ---- 上传 ----
    async function uploadFile(file, kbId, kbName) {
      if (!token.value) { say("请先填写上传令牌", "err"); return; }
      const target = (kbId || "").trim() || "uploads";
      uploading.value = true;
      task.value = { filename: file.name, status: "上传中", progress: 0 };
      try {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("kb_id", target);
        if (kbName) fd.append("kb_name", kbName);

        const res = await api("/kb/upload", { method: "POST", body: fd, token: token.value, raw: true });
        task.value = { ...res, status: res.status, progress: 0 };
        say(`已受理：${file.name}，正在后台建索引…`);
        await pollTask(res.task_id);
        await loadKbs();
        await loadFiles(target);
      } catch (e) {
        task.value = null;
        say("上传失败：" + e.message, "err");
      } finally {
        uploading.value = false;
      }
    }

    async function pollTask(taskId) {
      for (let i = 0; i < 300; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        let t;
        try { t = await api(`/kb/tasks/${taskId}`); }
        catch (e) { say("查询任务失败：" + e.message, "err"); return; }
        task.value = { ...task.value, ...t };
        if (t.status === "done" || t.status === "failed") {
          if (t.status === "done") say(`学习完成：${t.filename} → ${t.chunks} 个片段`);
          else say("建索引失败：" + (t.error || "未知错误"), "err");
          return;
        }
      }
      say("任务超时，请稍后在知识库里确认", "err");
    }

    function onDrop(e) {
      dragOver.value = false;
      const f = e.dataTransfer.files[0];
      if (f) uploadFile(f, activeKb.value === "rujia" ? "uploads" : activeKb.value, "");
    }
    function onPick(e) {
      const f = e.target.files[0];
      if (f) uploadFile(f, activeKb.value === "rujia" ? "uploads" : activeKb.value, "");
      e.target.value = "";
    }

    async function removeFile(fileId) {
      if (!confirm("删除该文件及其全部向量？")) return;
      try {
        await api(`/kb/${activeKb.value}/files/${fileId}`, { method: "DELETE", token: token.value });
        say("已删除");
        await loadFiles(activeKb.value);
        await loadKbs();
      } catch (e) { say("删除失败：" + e.message, "err"); }
    }

    async function removeKb(kbId) {
      if (!confirm(`删除整个知识库「${kbId}」？其中所有文件与向量都会清除。`)) return;
      try {
        await api(`/kb/${kbId}`, { method: "DELETE", token: token.value });
        say("知识库已删除");
        activeKb.value = "rujia";
        files.value = [];
        await loadKbs();
      } catch (e) { say("删除失败：" + e.message, "err"); }
    }

    // ---- 会话与对话 ----
    async function loadSessions() {
      try {
        const d = await api("/sessions");
        sessions.value = d.sessions || [];
      } catch (e) { /* 数据库未就绪时忽略 */ }
    }

    async function newSession() {
      try {
        const d = await api("/sessions", { method: "POST" });
        await loadSessions();
        await openSession(d.id);
      } catch (e) { say("新建会话失败：" + e.message, "err"); }
    }

    async function openSession(id) {
      sessionId.value = id;
      try {
        const d = await api(`/sessions/${id}/messages`);
        messages.value = d.messages || [];
        scrollDown();
      } catch (e) { say("读取会话失败：" + e.message, "err"); }
    }

    function scrollDown() {
      nextTick(() => {
        if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight;
      });
    }

    async function send() {
      const text = question.value.trim();
      if (!text) return;
      if (!sessionId.value) await newSession();
      if (!sessionId.value) return;
      busy.value = true;
      messages.value.push({ role: "user", content: text });
      question.value = "";
      scrollDown();
      try {
        const payload = { session_id: sessionId.value, message: text };
        if (activeKb.value) payload.kb_id = activeKb.value;
        const d = await api("/chat", { method: "POST", body: payload });
        messages.value.push({ role: "ai", content: d.answer });
      } catch (e) {
        messages.value.push({ role: "ai", content: "（出错：" + e.message + "）" });
      } finally {
        busy.value = false;
        scrollDown();
        loadSessions();
      }
    }

    onMounted(async () => {
      try { healthy.value = (await api("/health")).status === "ok"; }
      catch { healthy.value = false; }
      await loadKbs();
      await loadSessions();
      if (sessions.value.length) await openSession(sessions.value[0].id);
    });

    return {
      healthy, token, saveToken, kbs, activeKb, activeKbName, files, sessions,
      sessionId, messages, question, busy, uploading, dragOver, toast, task, chatBox,
      loadFiles, uploadFile, onDrop, onPick, removeFile, removeKb,
      newSession, openSession, send,
    };
  },
  template: `
  <header>
    <h1>儒家经典知识库</h1>
    <span class="sub">
      后端
      <template v-if="healthy === null">检测中…</template>
      <template v-else-if="healthy">✅ 正常</template>
      <template v-else>❌ 不可用</template>
    </span>
    <span style="flex:1"></span>
    <span v-if="toast.text" :class="toast.kind === 'err' ? 'err' : 'ok'">{{ toast.text }}</span>
  </header>

  <div class="wrap">
    <!-- 左栏 -->
    <div class="stack">
      <div class="card">
        <h2>上传令牌</h2>
        <div class="stack">
          <input v-model="token" type="password" placeholder="X-Upload-Token" @keyup.enter="saveToken">
          <button @click="saveToken">保存到本机</button>
          <div class="hint">保存在浏览器 localStorage，不会发给除本服务外的任何人。</div>
        </div>
      </div>

      <div class="card">
        <h2>知识库</h2>
        <div v-for="k in kbs" :key="k.id" class="kb" :class="{ active: k.id === activeKb }"
             @click="loadFiles(k.id)">
          <div class="name">
            {{ k.name }}
            <span v-if="k.builtin" class="tag builtin">内置</span>
          </div>
          <div class="meta">{{ k.id }} · {{ k.chunk_count }} 片段<span v-if="!k.builtin"> · {{ k.file_count }} 文件</span></div>
        </div>
        <div class="hint">点选后，右侧对话只查该库。内置库为只读。</div>
      </div>

      <div class="card">
        <h2>上传文档</h2>
        <div class="drop" :class="{ over: dragOver }"
             @dragover.prevent="dragOver = true"
             @dragleave="dragOver = false"
             @drop.prevent="onDrop"
             @click="$refs.picker.click()">
          <div v-if="!uploading">拖文件到这里，或点击选择</div>
          <div v-else>上传中…</div>
          <div class="hint" style="margin-top:6px">支持 docx / pdf / txt / md / xlsx / csv，单文件 ≤ 20MB</div>
        </div>
        <input ref="picker" type="file" style="display:none"
               accept=".docx,.pdf,.txt,.md,.xlsx,.csv" @change="onPick">

        <div v-if="task" style="margin-top:12px">
          <div class="hint">{{ task.filename }} · {{ task.status }}
            <template v-if="task.chunks"> · {{ task.chunks }} 片段</template>
          </div>
          <div class="bar"><i :style="{ width: task.status === 'done' ? '100%' : '60%' }"></i></div>
          <div v-if="task.error" class="err">{{ task.error }}</div>
        </div>
        <div class="hint" style="margin-top:8px">
          未选择知识库时，文件会进入 <code>uploads</code> 库。
        </div>
      </div>

      <div class="card" v-if="activeKb !== 'rujia'">
        <h2>{{ activeKbName }} · 文件</h2>
        <div v-if="!files.length" class="hint">暂无文件</div>
        <div v-for="f in files" :key="f.file_id" class="file">
          <span :title="f.filename">{{ f.filename }} <span class="tag">{{ f.chunks }} 片段</span></span>
          <button class="danger" @click="removeFile(f.file_id)">删除</button>
        </div>
        <button v-if="files.length" class="danger" style="margin-top:10px"
                @click="removeKb(activeKb)">删除整个知识库</button>
      </div>

      <div class="card">
        <h2>会话</h2>
        <div class="row" style="margin-bottom:10px">
          <button class="primary" @click="newSession">新建对话</button>
        </div>
        <div v-for="s in sessions" :key="s.id" class="kb" :class="{ active: s.id === sessionId }"
             @click="openSession(s.id)">
          <div class="name">{{ s.title }}</div>
          <div class="meta">{{ s.message_count }} 条</div>
        </div>
      </div>
    </div>

    <!-- 右栏 -->
    <div class="card">
      <h2>对话 · 当前知识库：{{ activeKbName }}</h2>
      <div class="log" ref="chatBox">
        <div v-if="!messages.length" class="hint">还没有消息，问点什么吧。</div>
        <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role === 'user' ? 'user' : 'ai'">
          <div class="who">{{ m.role === 'user' ? '你' : '助手' }}</div>
          <div>{{ m.content }}</div>
        </div>
      </div>
      <div class="row" style="margin-top:12px">
        <textarea v-model="question" rows="2" placeholder="输入问题，Enter 发送（Shift+Enter 换行）"
                  @keydown.enter.exact.prevent="send" :disabled="busy"></textarea>
        <button class="primary" :disabled="busy || !question.trim()" @click="send">
          {{ busy ? '思考中…' : '发送' }}
        </button>
      </div>
    </div>
  </div>
  `,
}).mount("#app");

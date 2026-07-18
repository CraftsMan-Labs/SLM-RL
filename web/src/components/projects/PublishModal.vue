<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiModal from '@/components/ui/UiModal.vue'

const props = defineProps<{
  open: boolean
  projectName: string
  hfUsername: string | null
  busy?: boolean
}>()

const emit = defineEmits<{
  close: []
  confirm: [repoName: string]
}>()

const REPO_SLUG_RE = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/

const repoName = ref('')

watch(
  () => props.open,
  (open) => {
    if (open) {
      repoName.value = `slm-rl-${props.projectName}`
    }
  },
)

const slug = computed(() => {
  let raw = repoName.value.trim()
  if (raw.includes('/')) {
    raw = raw.replace(/\/+$/, '').split('/').pop() || ''
  }
  return raw
})

const slugValid = computed(() => {
  const s = slug.value
  if (!s) return false
  if (s.endsWith('-data')) return false
  return REPO_SLUG_RE.test(s)
})

const owner = computed(() => props.hfUsername?.trim() || 'your-username')

const modelRepo = computed(() => `${owner.value}/${slug.value || '…'}`)
const datasetRepo = computed(() => `${owner.value}/${slug.value || '…'}-data`)

const hint = computed(() => {
  if (!slug.value) return 'Required — becomes the model repo under your HF account.'
  if (slug.value.endsWith('-data')) {
    return 'Use the model name only — we publish the dataset as <name>-data.'
  }
  if (!slugValid.value) {
    return 'Letters, digits, ., _, or - only (1–96 chars). No spaces.'
  }
  return 'Model + dataset upload to your account with the token from Welcome.'
})

function onSubmit() {
  if (!slugValid.value || props.busy) return
  emit('confirm', slug.value)
}
</script>

<template>
  <UiModal
    :open="open"
    eyebrow="Hugging Face"
    title="Publish project"
    @close="emit('close')"
  >
    <p class="lead">
      Uploads the champion LoRA (model) and training generations (dataset) to
      <strong>your</strong> Hugging Face account using the write token from Welcome.
    </p>

    <form class="form" @submit.prevent="onSubmit">
      <UiField
        label="Repo name"
        for-id="publish-repo-name"
        :hint="hint"
      >
        <input
          id="publish-repo-name"
          v-model="repoName"
          type="text"
          autocomplete="off"
          spellcheck="false"
          placeholder="e.g. slm-rl-my-boxing"
          :disabled="busy"
          :aria-invalid="Boolean(slug && !slugValid)"
        />
      </UiField>

      <dl class="preview">
        <div>
          <dt>Model</dt>
          <dd>
            <a
              v-if="slugValid && hfUsername"
              :href="`https://huggingface.co/${modelRepo}`"
              target="_blank"
              rel="noopener"
            >{{ modelRepo }}</a>
            <span v-else>{{ modelRepo }}</span>
          </dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>
            <a
              v-if="slugValid && hfUsername"
              :href="`https://huggingface.co/datasets/${datasetRepo}`"
              target="_blank"
              rel="noopener"
            >{{ datasetRepo }}</a>
            <span v-else>{{ datasetRepo }}</span>
          </dd>
        </div>
      </dl>

      <div class="actions">
        <UiButton type="button" variant="ghost" :disabled="busy" @click="emit('close')">
          Cancel
        </UiButton>
        <UiButton type="submit" :disabled="busy || !slugValid">
          {{ busy ? 'Publishing…' : 'Publish both' }}
        </UiButton>
      </div>
    </form>
  </UiModal>
</template>

<style scoped>
.lead {
  margin: 0;
  font-size: var(--text-sm);
  color: var(--muted);
  line-height: 1.45;
}

.lead strong {
  color: var(--text);
  font-weight: 600;
}

.form {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.preview {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  background: var(--surface-2, rgba(255, 255, 255, 0.04));
  border-radius: var(--radius-sm);
}

.preview > div {
  display: grid;
  grid-template-columns: 4.5rem 1fr;
  gap: var(--space-2);
  align-items: baseline;
}

dt {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

dd {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 13px;
  word-break: break-all;
}

dd a {
  color: var(--accent);
}

.actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-3);
}
</style>

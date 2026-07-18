<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import UiButton from '@/components/ui/UiButton.vue'
import UiCard from '@/components/ui/UiCard.vue'
import UiField from '@/components/ui/UiField.vue'
import { useProfile } from '@/composables/useProfile'

const router = useRouter()
const { save, loading, error, isOnboarded } = useProfile()

const name = ref('')
const token = ref('')
const localError = ref<string | null>(null)

const canSubmit = computed(() => name.value.trim().length > 0)

async function persist(withToken: boolean) {
  localError.value = null
  if (!name.value.trim()) {
    localError.value = withToken
      ? 'Enter a display name.'
      : 'Name is required, even to skip the token.'
    return
  }
  const trimmed = token.value.trim()
  if (withToken && trimmed && !trimmed.startsWith('hf_')) {
    localError.value = 'Paste a Hugging Face token starting with hf_.'
    return
  }
  try {
    await save(name.value.trim(), withToken ? trimmed || null : null)
    await router.push({ name: 'projects' })
  } catch {
    /* error surfaced via useProfile */
  }
}

async function onSubmit() {
  await persist(true)
}

async function onSkip() {
  await persist(false)
}
</script>

<template>
  <div class="welcome">
    <!-- 1. Hero -->
    <section class="hero" aria-label="Welcome">
      <div class="glow" aria-hidden="true" />
      <p class="eyebrow">SLM-RL · workshop</p>
      <h1>Hey, welcome to the platform.</h1>
      <p class="lead">
        This is what we’re going to do: teach a small language model to play a game,
        improve it with its own experience, and publish the result.
      </p>
      <div class="hero-actions">
        <a class="scroll-cta" href="#journey">
          <UiButton>See the journey</UiButton>
        </a>
        <RouterLink v-if="isOnboarded" class="continue" to="/projects">Go to projects →</RouterLink>
      </div>
    </section>

    <!-- 2. What we'll do -->
    <section id="journey" class="section journey">
      <p class="eyebrow">The loop</p>
      <h2>Four moves. One generation at a time.</h2>
      <ol class="steps">
        <li>
          <span class="num">01</span>
          <div>
            <h3>Play</h3>
            <p>Your model rolls out episodes in a text-native Atari game — Boxing, Space Invaders, and more.</p>
          </div>
        </li>
        <li>
          <span class="num">02</span>
          <div>
            <h3>Train</h3>
            <p>We turn those episodes into a dataset and fine-tune. Better play only sticks if eval says so.</p>
          </div>
        </li>
        <li>
          <span class="num">03</span>
          <div>
            <h3>Compare</h3>
            <p>Theater shows base vs champion on the same seeds — the workshop money shot.</p>
          </div>
        </li>
        <li>
          <span class="num">04</span>
          <div>
            <h3>Publish</h3>
            <p>Push datasets and adapters to your Hugging Face account when you’re ready to share.</p>
          </div>
        </li>
      </ol>
      <a class="scroll-cta mid" href="#credentials">
        <UiButton variant="secondary">Continue to credentials</UiButton>
      </a>
    </section>

    <!-- 3. HF credentials -->
    <section id="credentials" class="section credentials">
      <div class="cred-grid">
        <div class="copy">
          <p class="eyebrow">Step two</p>
          <h2>Connect Hugging Face</h2>
          <p class="lead-sm">
            Optional, but good to have — a write token unlocks publish.
            Skip anytime; rollouts, evolve, and theater work without it.
            We store it only on this machine — never in the repo.
          </p>
          <ol class="hf-steps">
            <li>
              Create an account at
              <a href="https://huggingface.co/join" target="_blank" rel="noopener">huggingface.co/join</a>
            </li>
            <li>
              Open
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Settings → Access Tokens</a>
              and create a <strong>write</strong> token
            </li>
            <li>Paste the token below with your display name — or skip</li>
          </ol>
        </div>

        <UiCard eyebrow="Credentials" title="Enter the workshop">
          <form class="form" @submit.prevent="onSubmit">
            <UiField label="Display name" for-id="name" hint="Shown in the header. Required.">
              <input id="name" v-model="name" autocomplete="nickname" placeholder="e.g. Ada" />
            </UiField>
            <UiField
              label="Hugging Face API key"
              for-id="token"
              hint="Starts with hf_. Optional — skip disables publish only."
            >
              <input
                id="token"
                v-model="token"
                type="password"
                autocomplete="off"
                placeholder="hf_..."
              />
            </UiField>
            <p v-if="localError || error" class="err" role="alert">
              {{ localError || error }}
            </p>
            <UiButton type="submit" :disabled="!canSubmit || loading" block>
              {{ loading ? 'Saving…' : 'Save and continue' }}
            </UiButton>
            <UiButton
              type="button"
              variant="secondary"
              :disabled="!canSubmit || loading"
              block
              @click="onSkip"
            >
              Skip for now
            </UiButton>
          </form>
        </UiCard>
      </div>
    </section>
  </div>
</template>

<style scoped>
.welcome {
  overflow-x: hidden;
}

.hero {
  position: relative;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  width: min(100% - 2 * var(--space-6), 920px);
  margin: 0 auto;
  padding: var(--space-12) 0;
}

.glow {
  position: absolute;
  inset: 10% auto auto 50%;
  width: min(70vw, 640px);
  height: 320px;
  transform: translateX(-50%);
  background: radial-gradient(circle, rgba(0, 153, 255, 0.18), transparent 70%);
  pointer-events: none;
  z-index: 0;
}

.hero > *:not(.glow) {
  position: relative;
  z-index: 1;
}

.eyebrow {
  margin: 0 0 var(--space-4);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}

h1 {
  max-width: 14ch;
  font-size: var(--text-4xl);
  margin-bottom: var(--space-6);
}

.lead {
  max-width: 38ch;
  font-size: var(--text-lg);
  color: var(--fg-2);
  line-height: 1.45;
  margin-bottom: var(--space-8);
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-4);
}

.scroll-cta {
  text-decoration: none;
}

.continue {
  color: var(--muted);
  font-size: var(--text-sm);
}

.section {
  width: min(100% - 2 * var(--space-6), var(--container-max));
  margin: 0 auto;
  padding: var(--section-y) 0;
}

.journey h2 {
  font-size: var(--text-3xl);
  max-width: 16ch;
  margin-bottom: var(--space-8);
}

.steps {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: var(--space-5);
}

.steps li {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: var(--space-5);
  padding: var(--space-5) 0;
  box-shadow: 0 1px 0 var(--border-soft);
}

.num {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  color: var(--accent);
  padding-top: 4px;
}

.steps h3 {
  font-size: var(--text-xl);
  margin-bottom: var(--space-2);
}

.steps p {
  color: var(--fg-2);
  max-width: 52ch;
}

.mid {
  display: inline-block;
  margin-top: var(--space-8);
}

.cred-grid {
  display: grid;
  grid-template-columns: 1.05fr 0.95fr;
  gap: var(--space-8);
  align-items: start;
}

.copy h2 {
  font-size: var(--text-3xl);
  max-width: 12ch;
  margin-bottom: var(--space-5);
}

.lead-sm {
  color: var(--fg-2);
  max-width: 40ch;
  margin-bottom: var(--space-6);
}

.hf-steps {
  margin: 0;
  padding-left: 1.2rem;
  color: var(--muted);
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.hf-steps strong {
  color: var(--fg);
  font-weight: 600;
}

.form {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.err {
  color: var(--danger);
  font-size: var(--text-sm);
}

@media (max-width: 809px) {
  .hero,
  .section {
    width: min(100% - 2 * var(--space-4), var(--container-max));
  }

  .cred-grid {
    grid-template-columns: 1fr;
  }

  h1 {
    max-width: none;
  }
}
</style>

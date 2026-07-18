<script setup lang="ts">
import { computed } from 'vue'
import { theaterFramesUrl, watchFramesUrl, type TheaterSide } from '@/api/watch'
import UiButton from '@/components/ui/UiButton.vue'
import { useFrameStream } from '@/composables/useFrameStream'

const props = withDefaults(
  defineProps<{
    projectName: string
    episodeId: string
    gen?: number | null
    /** Increments on every Watch click — restarts the multipart stream from the start. */
    restartToken?: number
    /** Inline stage for the rollout phase guide. */
    embedded?: boolean
    /** Optional HUD line under the title (e.g. last action). */
    hud?: string | null
    /** When set, stream from /theater/<name>/<side>/frames instead of /watch. */
    theaterSide?: TheaterSide | null
  }>(),
  {
    embedded: false,
    hud: null,
    theaterSide: null,
  },
)

const emit = defineEmits<{
  close: []
}>()

const streamUrl = computed(() => {
  if (!props.projectName || !props.episodeId) return null
  const token = props.restartToken ?? Date.now()
  if (props.theaterSide) {
    return theaterFramesUrl(
      props.projectName,
      props.theaterSide,
      props.episodeId,
      props.gen,
      token,
    )
  }
  return watchFramesUrl(props.projectName, props.episodeId, props.gen, token)
})

const { frameSrc, loading, error, stop } = useFrameStream(streamUrl)

function close() {
  // Drop the ALE stream before the parent clears episode state / unmounts.
  stop()
  emit('close')
}
</script>

<template>
  <aside
    class="panel"
    :class="{ embedded }"
    :aria-label="embedded ? 'Live rollout screen' : 'Live game screen'"
  >
    <header class="head">
      <div class="head-copy">
        <span class="title">
          {{ embedded ? 'Live play' : 'episode' }} {{ episodeId }}
        </span>
        <span v-if="hud" class="hud">{{ hud }}</span>
      </div>
      <UiButton variant="ghost" class="close" @click="close">Close</UiButton>
    </header>
    <div class="stage">
      <img
        v-if="frameSrc"
        class="frame"
        :src="frameSrc"
        alt="Game screen"
      />
      <p v-if="loading && !error" class="overlay" role="status">
        {{ embedded ? 'Catching up to live frame…' : 'Starting replay…' }}
      </p>
      <p v-else-if="!frameSrc && !error" class="overlay muted" role="status">
        Waiting for frames…
      </p>
    </div>
    <p v-if="error" class="msg" role="status">{{ error }}</p>
  </aside>
</template>

<style scoped>
.panel {
  position: sticky;
  top: 72px;
  padding: var(--space-4);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: var(--elev-raised);
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.panel.embedded {
  position: static;
  padding: 0;
  background: transparent;
  box-shadow: none;
  gap: var(--space-2);
  margin-top: var(--space-3);
}

.head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--space-3);
}

.head-copy {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.title {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--muted);
  letter-spacing: 0.04em;
}

.hud {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  color: var(--fg);
  letter-spacing: 0.02em;
}

.close {
  min-height: 32px;
  padding: 4px 12px;
  font-size: var(--text-xs);
}

.stage {
  position: relative;
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: #0a0a0a;
  box-shadow: 0 0 0 1px var(--border-soft);
  aspect-ratio: 4 / 3;
  width: 100%;
}

.panel.embedded .stage {
  max-width: 420px;
}

.frame {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  image-rendering: pixelated;
  background: #000;
}

.overlay {
  position: absolute;
  inset: 0;
  margin: 0;
  display: grid;
  place-items: center;
  padding: var(--space-3);
  text-align: center;
  font-size: var(--text-xs);
  color: var(--warn);
  line-height: 1.4;
  pointer-events: none;
  background: rgba(0, 0, 0, 0.35);
}

.overlay.muted {
  color: var(--muted);
}

.msg {
  margin: 0;
  font-size: var(--text-xs);
  color: var(--warn);
  line-height: 1.4;
}

@media (max-width: 809px) {
  .panel {
    position: static;
  }

  .panel.embedded .stage {
    max-width: none;
  }
}
</style>

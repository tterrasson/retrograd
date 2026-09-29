<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useArtifacts, useDownloadArtifact } from '@/api/artifacts'
import { useNotify } from '@/composables/useNotify'
import { ArtifactTable } from '@/features/artifacts'
import { ProblemAlert } from '@/features/shared'
import { useRunContext } from './context'

const run = useRunContext()
const router = useRouter()
const notify = useNotify()
const id = computed(() => run.value?.id ?? '')
const listing = useArtifacts(id)
const download = useDownloadArtifact(id)
const downloading = ref<string | null>(null)

async function start(name: string) {
  downloading.value = name
  try {
    await download.mutateAsync(name)
  } catch (error) {
    notify.error(error, `Download ${name}`)
  } finally {
    downloading.value = null
  }
}
</script>

<template>
  <div>
    <ProblemAlert :error="listing.error.value" />
    <v-card>
      <ArtifactTable
        :artifacts="listing.data.value?.artifacts ?? []"
        :loading="listing.isFetching.value"
        :downloading="downloading"
        @download="start"
        @checkpoints="router.push({ name: 'run-checkpoints', params: { id } })"
      />
    </v-card>
    <p class="text-caption rg-muted mt-2">
      A download goes through a signed link valid for a minute; the browser saves the file as it
      arrives.
    </p>
  </div>
</template>

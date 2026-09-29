<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { mdiEye, mdiEyeOff, mdiKeyVariant } from '@mdi/js'
import { useSessionStore } from '@/stores/session'
import { ProblemAlert } from '@/features/shared'

const session = useSessionStore()
const route = useRoute()
const router = useRouter()

const token = ref('')
const remember = ref(false)
const reveal = ref(false)
const busy = ref(false)
const refused = ref(false)
const failure = ref<unknown>(null)

async function submit() {
  if (!token.value.trim()) return
  busy.value = true
  refused.value = false
  failure.value = null
  try {
    const accepted = await session.login(token.value.trim(), remember.value)
    if (!accepted) {
      refused.value = true
      return
    }
    const next =
      typeof route.query.next === 'string' && route.query.next.startsWith('/')
        ? route.query.next
        : '/'
    await router.replace(next)
  } catch (error) {
    failure.value = error
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <v-container class="d-flex justify-center align-center" style="min-height: 80vh">
    <v-card width="440" class="pa-2">
      <v-card-title class="d-flex align-center ga-2">
        <v-icon :icon="mdiKeyVariant" />
        <h1 class="text-h6">Connect to retrograd</h1>
      </v-card-title>
      <v-card-text>
        <p class="text-body-2 mb-4">
          This server asks for the bearer token set in its configuration.
        </p>
        <form @submit.prevent="submit">
          <v-text-field
            v-model="token"
            label="Token"
            :type="reveal ? 'text' : 'password'"
            autocomplete="current-password"
            autofocus
            :append-inner-icon="reveal ? mdiEyeOff : mdiEye"
            :error-messages="refused ? ['The server refused this token.'] : []"
            @click:append-inner="reveal = !reveal"
          />
          <v-checkbox
            v-model="remember"
            label="Remember on this browser"
            density="compact"
            hide-details
          />
          <v-alert v-if="remember" type="warning" variant="tonal" density="compact" class="my-2">
            The token gives full control of the server. Kept in this browser's local storage, it
            stays until you sign out.
          </v-alert>
          <ProblemAlert :error="failure" />
          <v-btn
            type="submit"
            color="primary"
            block
            class="mt-3"
            :loading="busy"
            :disabled="!token.trim()"
          >
            Connect
          </v-btn>
        </form>
      </v-card-text>
    </v-card>
  </v-container>
</template>

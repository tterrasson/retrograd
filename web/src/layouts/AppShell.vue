<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  mdiBrightness6,
  mdiCompareHorizontal,
  mdiDatabaseOutline,
  mdiKeyboardOutline,
  mdiLogout,
  mdiMenu,
  mdiPlusCircleOutline,
  mdiServerNetwork,
  mdiViewList,
} from '@mdi/js'
import { useCapabilities, useHealth } from '@/api/discovery'
import { useSessionStore } from '@/stores/session'
import { usePreferencesStore, type ThemePreference } from '@/stores/preferences'
import { useShortcuts } from '@/composables/useShortcuts'
import { formatBytes } from '@/utils/format'

const session = useSessionStore()
const preferences = usePreferencesStore()
const route = useRoute()
const router = useRouter()

const ready = computed(() => session.authenticated)
const capabilities = useCapabilities(() => session.authenticated)
const health = useHealth()
const features = computed(() => capabilities.data.value?.features ?? session.capabilities?.features)
const viewer = computed(() => session.viewer)
const bare = computed(() => route.meta.public === true || !ready.value)
const drawer = ref<boolean | null>(null)
const help = ref(false)

const nav = [
  { title: 'Runs', icon: mdiViewList, to: '/runs' },
  { title: 'New run', icon: mdiPlusCircleOutline, to: '/runs/new' },
  { title: 'Datasets', icon: mdiDatabaseOutline, to: '/datasets' },
  { title: 'Compare', icon: mdiCompareHorizontal, to: '/compare' },
]

const themes: { value: ThemePreference; title: string }[] = [
  { value: 'system', title: 'System' },
  { value: 'light', title: 'Light' },
  { value: 'dark', title: 'Dark' },
]

function logout() {
  session.forget()
  void router.push({ name: 'login' })
}

useShortcuts({
  '?': () => (help.value = !help.value),
})

const shortcuts = [
  { keys: ['?'], text: 'show or hide this help' },
  { keys: ['j', '↓'], text: 'trajectories: next update' },
  { keys: ['k', '↑'], text: 'trajectories: previous update' },
  { keys: ['n'], text: 'trajectories: next member' },
  { keys: ['p'], text: 'trajectories: previous member' },
  { keys: ['c'], text: 'trajectories: add the open member to the comparison' },
  { keys: ['f'], text: 'trajectories: follow the latest update' },
]
</script>

<template>
  <template v-if="bare">
    <v-main>
      <v-alert v-if="session.unreachable" type="error" variant="tonal" class="ma-4" role="alert">
        The server did not answer: {{ session.unreachable }}
      </v-alert>
      <slot />
    </v-main>
  </template>
  <template v-else>
    <v-navigation-drawer v-if="!viewer" v-model="drawer" :width="220" aria-label="Main navigation">
      <v-list nav density="comfortable">
        <v-list-item
          v-for="item in nav"
          :key="item.to"
          :to="item.to"
          :prepend-icon="item.icon"
          :title="item.title"
          :exact="item.to === '/runs'"
          :active="
            item.to === '/runs'
              ? route.path === '/runs' ||
                (route.path.startsWith('/runs/') && route.path !== '/runs/new')
              : undefined
          "
        />
      </v-list>
    </v-navigation-drawer>

    <v-app-bar density="comfortable" flat border="b">
      <template v-if="!viewer" #prepend>
        <v-app-bar-nav-icon
          :icon="mdiMenu"
          aria-label="Toggle navigation"
          @click="drawer = !drawer"
        />
      </template>
      <v-app-bar-title>
        <router-link to="/" class="text-decoration-none" style="color: inherit"
          >retrograd</router-link
        >
        <v-chip v-if="viewer" class="ml-2" color="secondary">viewer</v-chip>
      </v-app-bar-title>

      <v-menu :close-on-content-click="false" location="bottom end">
        <template #activator="{ props: activator }">
          <v-btn
            v-bind="activator"
            variant="text"
            :prepend-icon="mdiServerNetwork"
            aria-label="Server status"
            :color="capabilities.isError.value ? 'error' : undefined"
          >
            <span class="d-none d-sm-inline">{{ health.data.value?.version ?? 'server' }}</span>
          </v-btn>
        </template>
        <v-card min-width="320" max-width="440">
          <v-card-title class="text-subtitle-1">Server</v-card-title>
          <v-card-text v-if="capabilities.data.value" class="text-body-2">
            <div v-if="health.data.value">version {{ health.data.value.version }}</div>
            <div>backends: {{ capabilities.data.value.backends.join(', ') || '–' }}</div>
            <div v-for="device in capabilities.data.value.devices" :key="device.name">
              {{ device.kind }} · {{ device.name }}
              <span v-if="device.total_bytes" class="rg-muted">
                ({{ formatBytes(device.free_bytes) }} free of {{ formatBytes(device.total_bytes) }})
              </span>
            </div>
            <div class="mt-2">
              budgets: VRAM {{ formatBytes(capabilities.data.value.budgets.vram.effective_bytes) }},
              RAM {{ formatBytes(capabilities.data.value.budgets.ram.effective_bytes) }}
              <span v-if="capabilities.data.value.budgets.unified">(unified)</span>
            </div>
            <div v-if="features" class="mt-2 d-flex flex-wrap ga-1">
              <v-chip>runs at once: {{ features.max_concurrent_runs }}</v-chip>
              <v-chip :color="features.serving_enabled ? 'success' : undefined">
                serving {{ features.serving_enabled ? 'on' : 'off' }}
              </v-chip>
              <v-chip>auth {{ features.auth ? 'on' : 'off' }}</v-chip>
              <v-chip>{{ features.mode }}</v-chip>
            </div>
          </v-card-text>
          <v-card-text v-else-if="capabilities.isError.value" class="text-error">
            {{ capabilities.error.value?.message }}
          </v-card-text>
        </v-card>
      </v-menu>

      <v-menu>
        <template #activator="{ props: activator }">
          <v-btn v-bind="activator" :icon="mdiBrightness6" aria-label="Theme" />
        </template>
        <v-list density="compact">
          <v-list-item
            v-for="option in themes"
            :key="option.value"
            :title="option.title"
            :active="preferences.theme === option.value"
            @click="preferences.theme = option.value"
          />
        </v-list>
      </v-menu>
      <v-btn :icon="mdiKeyboardOutline" aria-label="Keyboard shortcuts" @click="help = true" />
      <v-btn
        v-if="session.mode === 'token'"
        :icon="mdiLogout"
        aria-label="Forget the token"
        title="Forget the token"
        @click="logout"
      />
    </v-app-bar>

    <v-main>
      <v-container fluid class="pa-4">
        <slot />
      </v-container>
    </v-main>

    <v-dialog v-model="help" max-width="480">
      <v-card>
        <v-card-title>Keyboard shortcuts</v-card-title>
        <v-card-text>
          <v-table density="compact">
            <tbody>
              <tr v-for="item in shortcuts" :key="item.text">
                <td class="text-no-wrap">
                  <template v-for="(key, index) in item.keys" :key="key">
                    <span v-if="index" class="rg-muted"> or </span><kbd>{{ key }}</kbd>
                  </template>
                </td>
                <td>{{ item.text }}</td>
              </tr>
            </tbody>
          </v-table>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="help = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </template>
</template>

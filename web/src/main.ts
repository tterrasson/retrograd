import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { VueQueryPlugin } from '@tanstack/vue-query'
import App from './App.vue'
import { vuetify } from './plugins/vuetify'
import { createQueryClient, queryPluginOptions } from './plugins/query'
import { createAppRouter } from './router'
import { configureClient } from './api/client'
import { useSessionStore } from './stores/session'
import './styles/app.css'

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)

const router = createAppRouter()
const queryClient = createQueryClient()
const session = useSessionStore(pinia)

configureClient({
  token: () => session.token,
  onUnauthorized: () => {
    if (session.mode === 'open') return
    session.forget()
    queryClient.clear()
    const current = router.currentRoute.value
    if (current.name !== 'login') {
      void router.push({ name: 'login', query: { next: current.fullPath } })
    }
  },
})

app.use(router)
app.use(vuetify)
app.use(VueQueryPlugin, queryPluginOptions(queryClient))
app.mount('#app')

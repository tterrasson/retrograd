import { mount, type ComponentMountingOptions } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { QueryClient, VueQueryPlugin } from '@tanstack/vue-query'
import { createMemoryHistory, createRouter } from 'vue-router'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import type { Component } from 'vue'

/** Mounts a component with the plugins the application installs. */
export function mountWithPlugins<C extends Component>(
  component: C,
  options: ComponentMountingOptions<C> = {},
) {
  const vuetify = createVuetify({ components, directives })
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [{ path: '/:pathMatch(.*)*', component: { template: '<div />' } }],
  })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return mount(component, {
    ...options,
    global: {
      ...(options.global ?? {}),
      plugins: [vuetify, router, createPinia(), [VueQueryPlugin, { queryClient }]],
    },
  } as ComponentMountingOptions<C>)
}

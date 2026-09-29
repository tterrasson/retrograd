import { createRouter, createWebHistory, type RouteRecordRaw, type Router } from 'vue-router'
import { useSessionStore } from '@/stores/session'

/** The id of the only run a `viewer` process serves. */
export const VIEWER_RUN = 'local'

export const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'login',
    component: () => import('@/pages/LoginPage.vue'),
    meta: { public: true },
  },
  {
    path: '/',
    name: 'home',
    redirect: () => {
      const session = useSessionStore()
      return session.viewer ? `/runs/${VIEWER_RUN}/trajectories` : '/runs'
    },
  },
  { path: '/runs', name: 'runs', component: () => import('@/pages/RunsPage.vue') },
  { path: '/runs/new', name: 'new-run', component: () => import('@/pages/NewRunPage.vue') },
  {
    path: '/runs/:id',
    component: () => import('@/pages/RunPage.vue'),
    props: true,
    children: [
      { path: '', name: 'run', redirect: (to) => ({ name: 'run-overview', params: to.params }) },
      {
        path: 'overview',
        name: 'run-overview',
        component: () => import('@/pages/run/OverviewTab.vue'),
      },
      {
        path: 'metrics',
        name: 'run-metrics',
        component: () => import('@/pages/run/MetricsTab.vue'),
      },
      {
        path: 'journal',
        name: 'run-journal',
        component: () => import('@/pages/run/JournalTab.vue'),
      },
      {
        path: 'checkpoints',
        name: 'run-checkpoints',
        component: () => import('@/pages/run/CheckpointsTab.vue'),
      },
      {
        path: 'trajectories',
        name: 'run-trajectories',
        component: () => import('@/pages/run/TrajectoriesTab.vue'),
      },
      {
        path: 'artifacts',
        name: 'run-artifacts',
        component: () => import('@/pages/run/ArtifactsTab.vue'),
      },
      { path: 'config', name: 'run-config', component: () => import('@/pages/run/ConfigTab.vue') },
    ],
  },
  { path: '/datasets', name: 'datasets', component: () => import('@/pages/DatasetsPage.vue') },
  { path: '/compare', name: 'compare', component: () => import('@/pages/ComparePage.vue') },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/pages/NotFoundPage.vue'),
    meta: { public: true },
  },
]

export function createAppRouter(): Router {
  const router = createRouter({
    history: createWebHistory(),
    routes,
    scrollBehavior: (_to, _from, saved) => saved ?? undefined,
  })

  router.beforeEach(async (to) => {
    const session = useSessionStore()
    await session.init()
    if (session.unreachable) return true
    if (to.meta.public) {
      if (to.name === 'login' && session.authenticated) {
        const next = typeof to.query.next === 'string' ? to.query.next : '/'
        return next.startsWith('/') ? next : '/'
      }
      return true
    }
    if (!session.authenticated) {
      return { name: 'login', query: { next: to.fullPath } }
    }
    // A viewer process serves one run's trajectories and nothing else.
    if (session.viewer && !to.path.startsWith(`/runs/${VIEWER_RUN}/trajectories`)) {
      return `/runs/${VIEWER_RUN}/trajectories`
    }
    return true
  })

  return router
}

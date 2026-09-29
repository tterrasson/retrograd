import 'vuetify/styles'
import { createVuetify, type ThemeDefinition } from 'vuetify'
import { aliases, mdi } from 'vuetify/iconsets/mdi-svg'

// The palette of the documentation site: a violet brand drawn from the logo,
// amber for warnings, magenta for what matters, a rose for danger. Every text
// color keeps at least 4.5:1 on the surfaces it is drawn on.
const light: ThemeDefinition = {
  dark: false,
  colors: {
    background: '#fcfaff',
    surface: '#ffffff',
    'surface-bright': '#ffffff',
    'surface-light': '#f1ebf8',
    'surface-variant': '#4d4159',
    'on-surface-variant': '#f7f3fc',
    'on-background': '#221a2e',
    'on-surface': '#221a2e',
    primary: '#7a2fd4',
    'primary-darken-1': '#5c1fae',
    secondary: '#a8126f',
    'secondary-darken-1': '#8f0d5e',
    error: '#b4234a',
    info: '#5c1fae',
    success: '#1f7a4d',
    warning: '#8a560f',
  },
  variables: {
    'border-color': '#bda9d6',
    'border-opacity': 0.6,
    'medium-emphasis-opacity': 0.72,
    'code-background-color': '#ece3f7',
  },
}

const dark: ThemeDefinition = {
  dark: true,
  colors: {
    background: '#151020',
    surface: '#1d1630',
    'surface-bright': '#281f3d',
    'surface-light': '#281f3d',
    'surface-variant': '#cdc2dc',
    'on-surface-variant': '#1d1630',
    'on-background': '#f3eefa',
    'on-surface': '#f3eefa',
    primary: '#d4a4f4',
    'primary-darken-1': '#a874e8',
    secondary: '#f78fb0',
    'secondary-darken-1': '#d96d92',
    error: '#ff8fa3',
    info: '#e0bcf8',
    success: '#7fd4a6',
    warning: '#f9c676',
  },
  variables: {
    'border-color': '#524369',
    'border-opacity': 0.8,
    'medium-emphasis-opacity': 0.78,
    'code-background-color': '#0e0b15',
  },
}

export const vuetify = createVuetify({
  theme: {
    defaultTheme: 'retrogradLight',
    themes: { retrogradLight: light, retrogradDark: dark },
  },
  icons: { defaultSet: 'mdi', aliases, sets: { mdi } },
  defaults: {
    VBtn: { variant: 'flat' },
    VCard: { variant: 'flat', border: true },
    VTextField: { variant: 'outlined', density: 'comfortable' },
    VSelect: { variant: 'outlined', density: 'comfortable' },
    VAutocomplete: { variant: 'outlined', density: 'comfortable' },
    VTextarea: { variant: 'outlined', density: 'comfortable' },
    VDataTable: { density: 'comfortable', hover: true },
    VDataTableServer: { density: 'comfortable', hover: true },
    VChip: { size: 'small', variant: 'tonal' },
  },
})

export const THEMES = { light: 'retrogradLight', dark: 'retrogradDark' } as const

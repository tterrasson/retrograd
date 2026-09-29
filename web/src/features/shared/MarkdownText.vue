<script setup lang="ts">
import { computed } from 'vue'
import { renderMarkdown } from '@/utils/markdown'

const props = withDefaults(defineProps<{ text: string; markdown?: boolean }>(), { markdown: true })
// Sanitized by DOMPurify after markdown-it rendered it with raw HTML disabled.
const html = computed(() => (props.markdown ? renderMarkdown(props.text) : ''))
</script>

<template>
  <!-- eslint-disable-next-line vue/no-v-html -->
  <div v-if="markdown" class="rg-markdown" v-html="html" />
  <pre v-else class="text-pre">{{ text }}</pre>
</template>

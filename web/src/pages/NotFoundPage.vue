<script setup lang="ts">
import { mdiMapMarkerQuestionOutline } from '@mdi/js'
import { EmptyState } from '@/features/shared'
</script>

<template>
  <EmptyState
    :icon="mdiMapMarkerQuestionOutline"
    title="Nothing here"
    text="This address does not match any page."
  >
    <v-btn to="/" color="primary">Back to the start</v-btn>
  </EmptyState>
</template>

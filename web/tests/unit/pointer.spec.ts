import { describe, expect, it } from 'vitest'
import {
  getAt,
  hasAt,
  leafPaths,
  parsePointer,
  pathToPointer,
  pointerToPath,
  removeAt,
  setAt,
  toPointer,
} from '@/utils/pointer'

describe('JSON Pointer', () => {
  it('escapes and unescapes ~ and /', () => {
    expect(parsePointer('/a~1b/c~0d')).toEqual(['a/b', 'c~d'])
    expect(toPointer(['a/b', 'c~d'])).toBe('/a~1b/c~0d')
    expect(parsePointer('')).toEqual([])
  })

  it('converts between pointers and dotted paths under a prefix', () => {
    expect(pointerToPath('/params/training/lr', '/params')).toBe('training.lr')
    expect(pointerToPath('/recipe/model', '/params')).toBeNull()
    expect(pointerToPath('/paramsx/y', '/params')).toBeNull()
    expect(pathToPointer('training.lr', '/params')).toBe('/params/training/lr')
  })
})

describe('partial trees', () => {
  it('sets a value, creating the tables on the way, without touching the input', () => {
    const tree = { training: { epochs: 2 } }
    const next = setAt(tree, ['lora', 'rank'], 16)
    expect(next).toEqual({ training: { epochs: 2 }, lora: { rank: 16 } })
    expect(tree).toEqual({ training: { epochs: 2 } })
  })

  it('removes a value and the tables it leaves empty', () => {
    const tree = { training: { lr: 1e-4 }, lora: { rank: 8, alpha: 16 } }
    expect(removeAt(tree, ['training', 'lr'])).toEqual({ lora: { rank: 8, alpha: 16 } })
    expect(removeAt(tree, ['lora', 'rank'])).toEqual({
      training: { lr: 1e-4 },
      lora: { alpha: 16 },
    })
    expect(removeAt(tree, ['nope', 'x'])).toBe(tree)
  })

  it('reads and tests paths, and lists leaves', () => {
    const tree = { a: { b: [1, 2], c: null }, d: {} }
    expect(getAt(tree, ['a', 'b', '1'])).toBe(2)
    expect(hasAt(tree, ['a', 'c'])).toBe(true)
    expect(hasAt(tree, ['a', 'x'])).toBe(false)
    expect(leafPaths(tree)).toEqual([['a', 'b'], ['a', 'c'], ['d']])
  })
})

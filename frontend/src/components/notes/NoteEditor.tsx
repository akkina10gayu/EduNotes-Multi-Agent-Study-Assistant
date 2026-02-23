'use client'

import { useState, useReducer } from 'react'

interface NoteEditorProps {
  initialNotes: string
  onSave: (notes: string) => void
  onCancel: () => void
}

type Action = { type: 'edit'; value: string } | { type: 'undo' } | { type: 'redo' }
interface State { current: string; past: string[]; future: string[] }

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'edit':
      return { current: action.value, past: [...state.past, state.current], future: [] }
    case 'undo':
      if (state.past.length === 0) return state
      return { current: state.past[state.past.length - 1], past: state.past.slice(0, -1), future: [state.current, ...state.future] }
    case 'redo':
      if (state.future.length === 0) return state
      return { current: state.future[0], past: [...state.past, state.current], future: state.future.slice(1) }
  }
}

export default function NoteEditor({ initialNotes, onSave, onCancel }: NoteEditorProps) {
  const [state, dispatch] = useReducer(reducer, { current: initialNotes, past: [], future: [] })

  return (
    <div className="space-y-3">
      <textarea
        value={state.current}
        onChange={(e) => dispatch({ type: 'edit', value: e.target.value })}
        rows={20}
        className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white font-mono text-sm resize-y focus:outline-none focus:border-[#6CA0DC]"
      />
      <div className="flex gap-2">
        <button onClick={() => onSave(state.current)} className="px-4 py-1.5 text-sm bg-[#6CA0DC] text-white rounded-lg hover:bg-[#5a8ec4]">Save</button>
        <button onClick={() => dispatch({ type: 'undo' })} disabled={state.past.length === 0} className="px-3 py-1.5 text-sm bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 disabled:opacity-30">Undo</button>
        <button onClick={() => dispatch({ type: 'redo' })} disabled={state.future.length === 0} className="px-3 py-1.5 text-sm bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 disabled:opacity-30">Redo</button>
        <button onClick={onCancel} className="px-3 py-1.5 text-sm text-gray-400 hover:text-white">Cancel</button>
      </div>
    </div>
  )
}

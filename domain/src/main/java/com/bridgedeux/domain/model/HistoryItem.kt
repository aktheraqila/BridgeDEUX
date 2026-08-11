package com.bridgedeux.domain.model

data class HistoryItem(
    val id: Long,
    val sourceText: String,
    val translatedText: String,
    val sourceLanguage: Language,
    val targetLanguage: Language
)
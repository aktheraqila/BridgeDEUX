package com.bridgedeux.domain.model

data class TranslationRequest(
    val text: String,
    val sourceLanguage: Language,
    val targetLanguage: Language
)
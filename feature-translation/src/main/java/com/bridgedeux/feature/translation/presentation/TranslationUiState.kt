package com.bridgedeux.feature.translation.presentation

import com.bridgedeux.domain.model.Language

data class TranslationUiState(
    val inputText: String = "",
    val translatedText: String = "",
    val sourceLanguage: Language = Language.ENGLISH,
    val targetLanguage: Language = Language.GERMAN,
    val isLoading: Boolean = false,
    val errorMessage: String? = null
)
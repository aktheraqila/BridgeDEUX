package com.bridgedeux.feature.history.presentation

import java.util.Collections.emptyList

data class HistoryItemUi(
    val id: Long,
    val sourceText: String,
    val translatedText: String,
    val sourceLanguage: String,
    val targetLanguage: String
)

data class HistoryUiState(
    val searchQuery: String = "",
    val historyItems: List<HistoryItemUi> = emptyList(),
    val isLoading: Boolean = false,
    val errorMessage: String? = null
)
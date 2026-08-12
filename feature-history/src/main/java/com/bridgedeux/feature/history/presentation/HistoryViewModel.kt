package com.bridgedeux.feature.history.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.bridgedeux.domain.model.HistoryItem
import com.bridgedeux.domain.model.Language
import com.bridgedeux.domain.usecase.ClearHistoryUseCase
import com.bridgedeux.domain.usecase.DeleteHistoryUseCase
import com.bridgedeux.domain.usecase.GetHistoryUseCase
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class HistoryViewModel(
    private val getHistoryUseCase: GetHistoryUseCase,
    private val deleteHistoryUseCase: DeleteHistoryUseCase,
    private val clearHistoryUseCase: ClearHistoryUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(
        HistoryUiState()
    )

    val uiState: StateFlow<HistoryUiState> =
        _uiState.asStateFlow()

    init {
        observeHistory()
    }

    private fun observeHistory() {
        viewModelScope.launch {

            _uiState.value = _uiState.value.copy(
                isLoading = true,
                errorMessage = null
            )

            getHistoryUseCase()
                .catch { throwable ->

                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        errorMessage =
                            throwable.message ?: "Unable to load history."
                    )
                }
                .collect { history ->

                    _uiState.value = _uiState.value.copy(
                        historyItems = history.map { item ->
                            HistoryItemUi(
                                id = item.id,
                                sourceText = item.sourceText,
                                translatedText = item.translatedText,
                                sourceLanguage = item.sourceLanguage.name,
                                targetLanguage = item.targetLanguage.name
                            )
                        },
                        isLoading = false,
                        errorMessage = null
                    )
                }
        }
    }

    fun onSearchQueryChanged(
        query: String
    ) {
        _uiState.value = _uiState.value.copy(
            searchQuery = query
        )
    }

    fun onDeleteHistoryItem(
        item: HistoryItemUi
    ) {
        viewModelScope.launch {

            runCatching {
                deleteHistoryUseCase(
                    item.toDomainHistoryItem()
                )
            }.onFailure { throwable ->

                _uiState.value = _uiState.value.copy(
                    errorMessage =
                        throwable.message
                            ?: "Unable to delete history."
                )
            }
        }
    }

    fun onClearHistory() {
        viewModelScope.launch {

            runCatching {
                clearHistoryUseCase()
            }.onFailure { throwable ->

                _uiState.value = _uiState.value.copy(
                    errorMessage =
                        throwable.message
                            ?: "Unable to clear history."
                )
            }
        }
    }

    private fun HistoryItemUi.toDomainHistoryItem(): HistoryItem {
        return HistoryItem(
            id = id,
            sourceText = sourceText,
            translatedText = translatedText,
            sourceLanguage = Language.valueOf(sourceLanguage),
            targetLanguage = Language.valueOf(targetLanguage)
        )
    }
}
package com.bridgedeux.feature.translation.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.bridgedeux.core.result.AppError
import com.bridgedeux.core.result.AppResult
import com.bridgedeux.domain.model.HistoryItem
import com.bridgedeux.domain.usecase.SaveHistoryUseCase
import com.bridgedeux.domain.usecase.TranslateTextUseCase
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class TranslationViewModel(
    private val translateTextUseCase: TranslateTextUseCase,
    private val saveHistoryUseCase: SaveHistoryUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(
        TranslationUiState()
    )

    val uiState: StateFlow<TranslationUiState> =
        _uiState.asStateFlow()

    fun onInputTextChanged(
        text: String
    ) {
        _uiState.value = _uiState.value.copy(
            inputText = text,
            errorMessage = null
        )
    }

    fun onSwapLanguages() {
        val currentState = _uiState.value

        _uiState.value = currentState.copy(
            sourceLanguage = currentState.targetLanguage,
            targetLanguage = currentState.sourceLanguage,
            inputText = currentState.translatedText,
            translatedText = currentState.inputText,
            errorMessage = null
        )
    }

    fun onTranslateClicked() {
        val currentState = _uiState.value

        viewModelScope.launch {
            _uiState.value = currentState.copy(
                isLoading = true,
                errorMessage = null
            )

            when (
                val result = translateTextUseCase(
                    text = currentState.inputText,
                    sourceLanguage = currentState.sourceLanguage,
                    targetLanguage = currentState.targetLanguage
                )
            ) {
                is AppResult.Success -> {

                    val translationResult = result.data

                    _uiState.value = _uiState.value.copy(
                        translatedText = translationResult.translatedText,
                        isLoading = false,
                        errorMessage = null
                    )

                    saveHistoryUseCase(
                        HistoryItem(
                            id = 0L,
                            sourceText = translationResult.sourceText,
                            translatedText = translationResult.translatedText,
                            sourceLanguage = translationResult.sourceLanguage,
                            targetLanguage = translationResult.targetLanguage
                        )
                    )
                }

                is AppResult.Error -> {
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        errorMessage = result.error.toUserMessage()
                    )
                }
            }
        }
    }

    private fun AppError.toUserMessage(): String {
        return when (this) {
            AppError.Unknown -> {
                "An unknown error occurred."
            }

            is AppError.InvalidInput -> {
                reason
            }

            is AppError.ModelUnavailable -> {
                "Required model is unavailable: $modelId"
            }

            is AppError.InferenceFailure -> {
                message ?: "Translation inference failed."
            }
        }
    }
}
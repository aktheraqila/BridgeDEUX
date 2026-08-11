package com.bridgedeux.feature.translation.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.bridgedeux.domain.usecase.SaveHistoryUseCase
import com.bridgedeux.domain.usecase.TranslateTextUseCase

fun translationViewModelFactory(
    translateTextUseCase: TranslateTextUseCase,
    saveHistoryUseCase: SaveHistoryUseCase
): ViewModelProvider.Factory =
    object : ViewModelProvider.Factory {

        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(
            modelClass: Class<T>
        ): T {
            require(
                modelClass.isAssignableFrom(
                    TranslationViewModel::class.java
                )
            ) {
                "Unknown ViewModel class: ${modelClass.name}"
            }

            return TranslationViewModel(
                translateTextUseCase = translateTextUseCase,
                saveHistoryUseCase = saveHistoryUseCase
            ) as T
        }
    }
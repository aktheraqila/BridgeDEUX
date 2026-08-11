package com.bridgedeux.feature.history.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.bridgedeux.domain.usecase.ClearHistoryUseCase
import com.bridgedeux.domain.usecase.DeleteHistoryUseCase
import com.bridgedeux.domain.usecase.GetHistoryUseCase

fun historyViewModelFactory(
    getHistoryUseCase: GetHistoryUseCase,
    deleteHistoryUseCase: DeleteHistoryUseCase,
    clearHistoryUseCase: ClearHistoryUseCase
): ViewModelProvider.Factory =
    object : ViewModelProvider.Factory {

        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(
            modelClass: Class<T>
        ): T {

            require(
                modelClass.isAssignableFrom(
                    HistoryViewModel::class.java
                )
            ) {
                "Unknown ViewModel class: ${modelClass.name}"
            }

            return HistoryViewModel(
                getHistoryUseCase = getHistoryUseCase,
                deleteHistoryUseCase = deleteHistoryUseCase,
                clearHistoryUseCase = clearHistoryUseCase
            ) as T
        }
    }
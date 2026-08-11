package com.bridgedeux.domain.usecase

import com.bridgedeux.domain.repository.HistoryRepository

class ClearHistoryUseCase(
    private val historyRepository: HistoryRepository
) {
    suspend operator fun invoke() {
        historyRepository.clearHistory()
    }
}
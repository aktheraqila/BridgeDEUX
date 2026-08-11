package com.bridgedeux.domain.usecase

import com.bridgedeux.domain.model.HistoryItem
import com.bridgedeux.domain.repository.HistoryRepository

class SaveHistoryUseCase(
    private val historyRepository: HistoryRepository
) {
    suspend operator fun invoke(
        historyItem: HistoryItem
    ) {
        historyRepository.saveHistoryItem(historyItem)
    }
}
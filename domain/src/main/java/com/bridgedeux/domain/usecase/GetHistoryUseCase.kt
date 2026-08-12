package com.bridgedeux.domain.usecase

import com.bridgedeux.domain.model.HistoryItem
import com.bridgedeux.domain.repository.HistoryRepository
import kotlinx.coroutines.flow.Flow

class GetHistoryUseCase(
    private val historyRepository: HistoryRepository
) {

    operator fun invoke(): Flow<List<HistoryItem>> {
        return historyRepository.observeHistory()
    }
}
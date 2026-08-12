package com.bridgedeux.domain.repository

import com.bridgedeux.domain.model.HistoryItem
import kotlinx.coroutines.flow.Flow

interface HistoryRepository {

    suspend fun getHistory(): List<HistoryItem>

    fun observeHistory(): Flow<List<HistoryItem>>

    suspend fun saveHistoryItem(
        historyItem: HistoryItem
    )

    suspend fun deleteHistoryItem(
        historyItem: HistoryItem
    )

    suspend fun clearHistory()
}
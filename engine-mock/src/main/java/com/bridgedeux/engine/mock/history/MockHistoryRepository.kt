package com.bridgedeux.engine.mock.history

import com.bridgedeux.domain.model.HistoryItem
import com.bridgedeux.domain.model.Language
import com.bridgedeux.domain.repository.HistoryRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class MockHistoryRepository : HistoryRepository {

    override suspend fun getHistory(): List<HistoryItem> {
        delay(300)

        return mockHistory()
    }

    override fun observeHistory(): Flow<List<HistoryItem>> = flow {
        delay(300)
        emit(mockHistory())
    }

    override suspend fun saveHistoryItem(
        historyItem: HistoryItem
    ) {
        delay(100)
    }

    override suspend fun deleteHistoryItem(
        historyItem: HistoryItem
    ) {
        delay(100)
    }

    override suspend fun clearHistory() {
        delay(100)
    }

    private fun mockHistory(): List<HistoryItem> {
        return listOf(
            HistoryItem(
                id = 1L,
                sourceText = "Hello",
                translatedText = "Hallo",
                sourceLanguage = Language.ENGLISH,
                targetLanguage = Language.GERMAN
            ),
            HistoryItem(
                id = 2L,
                sourceText = "Good Morning",
                translatedText = "Guten Morgen",
                sourceLanguage = Language.ENGLISH,
                targetLanguage = Language.GERMAN
            ),
            HistoryItem(
                id = 3L,
                sourceText = "Danke",
                translatedText = "Thank you",
                sourceLanguage = Language.GERMAN,
                targetLanguage = Language.ENGLISH
            )
        )
    }
}
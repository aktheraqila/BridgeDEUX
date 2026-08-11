package com.bridgedeux.data.local.repository

import com.bridgedeux.data.local.database.TranslationDao
import com.bridgedeux.data.local.database.TranslationEntity
import com.bridgedeux.domain.model.HistoryItem
import com.bridgedeux.domain.model.Language
import com.bridgedeux.domain.repository.HistoryRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class LocalHistoryRepository(
    private val translationDao: TranslationDao
) : HistoryRepository {

    override suspend fun getHistory(): List<HistoryItem> {
        return translationDao
            .getTranslations()
            .map { it.toDomainModel() }
    }

    override fun observeHistory(): Flow<List<HistoryItem>> {
        return translationDao
            .observeTranslations()
            .map { translations ->
                translations.map { it.toDomainModel() }
            }
    }

    override suspend fun saveHistoryItem(
        historyItem: HistoryItem
    ) {
        translationDao.insertTranslation(
            historyItem.toEntity()
        )
    }

    override suspend fun deleteHistoryItem(
        historyItem: HistoryItem
    ) {
        translationDao.deleteTranslation(
            historyItem.toEntity()
        )
    }

    override suspend fun clearHistory() {
        translationDao.deleteAllTranslations()
    }
}

private fun TranslationEntity.toDomainModel(): HistoryItem {
    return HistoryItem(
        id = id,
        sourceText = sourceText,
        translatedText = translatedText,
        sourceLanguage = Language.entries.first {
            it.code == sourceLanguageCode
        },
        targetLanguage = Language.entries.first {
            it.code == targetLanguageCode
        }
    )
}

private fun HistoryItem.toEntity(): TranslationEntity {
    return TranslationEntity(
        id = id,
        sourceText = sourceText,
        translatedText = translatedText,
        sourceLanguageCode = sourceLanguage.code,
        targetLanguageCode = targetLanguage.code
    )
}
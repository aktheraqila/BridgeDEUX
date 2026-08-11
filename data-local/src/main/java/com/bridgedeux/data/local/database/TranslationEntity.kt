package com.bridgedeux.data.local.database

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(
    tableName = "translations"
)
data class TranslationEntity(

    @PrimaryKey(autoGenerate = true)
    val id: Long = 0L,

    val sourceText: String,

    val translatedText: String,

    val sourceLanguageCode: String,

    val targetLanguageCode: String
)
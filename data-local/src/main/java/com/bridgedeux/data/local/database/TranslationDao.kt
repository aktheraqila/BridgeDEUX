package com.bridgedeux.data.local.database

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

@Dao
interface TranslationDao {

    @Query(
        """
        SELECT * 
        FROM translations
        ORDER BY id DESC
        """
    )
    fun observeTranslations(): Flow<List<TranslationEntity>>

    @Query(
        """
        SELECT * 
        FROM translations
        ORDER BY id DESC
        """
    )
    suspend fun getTranslations(): List<TranslationEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTranslation(
        translation: TranslationEntity
    )

    @Upsert
    suspend fun upsertTranslation(
        translation: TranslationEntity
    )

    @Delete
    suspend fun deleteTranslation(
        translation: TranslationEntity
    )

    @Query("DELETE FROM translations")
    suspend fun deleteAllTranslations()
}
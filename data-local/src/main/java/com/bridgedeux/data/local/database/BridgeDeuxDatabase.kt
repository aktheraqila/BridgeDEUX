package com.bridgedeux.data.local.database

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(
    entities = [
        TranslationEntity::class
    ],
    version = 1,
    exportSchema = true
)
abstract class BridgeDeuxDatabase : RoomDatabase() {

    abstract fun translationDao(): TranslationDao
}
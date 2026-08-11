//package com.bridgedeux.data.local.repository
//
//import androidx.datastore.core.DataStore
//import androidx.datastore.preferences.core.Preferences
//import androidx.datastore.preferences.core.booleanPreferencesKey
//import androidx.datastore.preferences.core.edit
//import androidx.datastore.preferences.preferencesDataStoreFile
//import android.content.Context
//
//import com.bridgedeux.domain.repository.SettingsRepository
//
//import kotlinx.coroutines.flow.Flow
//import kotlinx.coroutines.flow.map
//
//class LocalSettingsRepository(
//    private val context: Context
//) : SettingsRepository {
//
//    private val dataStore: DataStore<Preferences> =
//        androidx.datastore.preferences.preferencesDataStoreFile(
//            "bridgedeux_settings"
//        ).let { file ->
//            androidx.datastore.preferences.preferencesDataStore(
//                file = file
//            )
//        }
//
//    private object Keys {
//        val developerModeEnabled =
//            booleanPreferencesKey("developer_mode_enabled")
//
//        val darkModeEnabled =
//            booleanPreferencesKey("dark_mode_enabled")
//
//        val voicePlaybackEnabled =
//            booleanPreferencesKey("voice_playback_enabled")
//    }
//
//    override fun observeDeveloperMode(): Flow<Boolean> =
//        dataStore.data.map { preferences ->
//            preferences[Keys.developerModeEnabled] ?: false
//        }
//
//    override suspend fun setDeveloperModeEnabled(
//        enabled: Boolean
//    ) {
//        dataStore.edit { preferences ->
//            preferences[Keys.developerModeEnabled] = enabled
//        }
//    }
//
//    override fun observeDarkMode(): Flow<Boolean> =
//        dataStore.data.map { preferences ->
//            preferences[Keys.darkModeEnabled] ?: false
//        }
//
//    override suspend fun setDarkModeEnabled(
//        enabled: Boolean
//    ) {
//        dataStore.edit { preferences ->
//            preferences[Keys.darkModeEnabled] = enabled
//        }
//    }
//
//    override fun observeVoicePlayback(): Flow<Boolean> =
//        dataStore.data.map { preferences ->
//            preferences[Keys.voicePlaybackEnabled] ?: true
//        }
//
//    override suspend fun setVoicePlaybackEnabled(
//        enabled: Boolean
//    ) {
//        dataStore.edit { preferences ->
//            preferences[Keys.voicePlaybackEnabled] = enabled
//        }
//    }
//}

package com.bridgedeux.data.local.repository

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit

import com.bridgedeux.domain.repository.SettingsRepository

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class LocalSettingsRepository(
    private val dataStore: DataStore<Preferences>
) : SettingsRepository {

    private object Keys {

        val developerModeEnabled =
            booleanPreferencesKey("developer_mode_enabled")

        val darkModeEnabled =
            booleanPreferencesKey("dark_mode_enabled")

        val voicePlaybackEnabled =
            booleanPreferencesKey("voice_playback_enabled")
    }

    override fun observeDeveloperMode(): Flow<Boolean> =
        dataStore.data.map { preferences ->
            preferences[Keys.developerModeEnabled] ?: false
        }

    override suspend fun setDeveloperModeEnabled(
        enabled: Boolean
    ) {
        dataStore.edit { preferences ->
            preferences[Keys.developerModeEnabled] = enabled
        }
    }

    override fun observeDarkMode(): Flow<Boolean> =
        dataStore.data.map { preferences ->
            preferences[Keys.darkModeEnabled] ?: false
        }

    override suspend fun setDarkModeEnabled(
        enabled: Boolean
    ) {
        dataStore.edit { preferences ->
            preferences[Keys.darkModeEnabled] = enabled
        }
    }

    override fun observeVoicePlayback(): Flow<Boolean> =
        dataStore.data.map { preferences ->
            preferences[Keys.voicePlaybackEnabled] ?: true
        }

    override suspend fun setVoicePlaybackEnabled(
        enabled: Boolean
    ) {
        dataStore.edit { preferences ->
            preferences[Keys.voicePlaybackEnabled] = enabled
        }
    }
}
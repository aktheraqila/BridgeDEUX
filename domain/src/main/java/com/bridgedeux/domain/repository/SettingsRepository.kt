package com.bridgedeux.domain.repository

import kotlinx.coroutines.flow.Flow

interface SettingsRepository {

    fun observeDeveloperMode(): Flow<Boolean>

    suspend fun setDeveloperModeEnabled(enabled: Boolean)

    fun observeDarkMode(): Flow<Boolean>

    suspend fun setDarkModeEnabled(enabled: Boolean)

    fun observeVoicePlayback(): Flow<Boolean>

    suspend fun setVoicePlaybackEnabled(enabled: Boolean)
}
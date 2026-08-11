package com.bridgedeux

import com.bridgedeux.data.local.repository.LocalSettingsRepository

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStoreFile
import androidx.datastore.preferences.core.PreferenceDataStoreFactory

import android.content.Context
import androidx.lifecycle.ViewModelProvider
import androidx.room.Room

import com.bridgedeux.data.local.database.BridgeDeuxDatabase
import com.bridgedeux.data.local.repository.LocalHistoryRepository

import com.bridgedeux.domain.usecase.GetHistoryUseCase
import com.bridgedeux.domain.usecase.SaveHistoryUseCase
import com.bridgedeux.domain.usecase.TranslateTextUseCase
import com.bridgedeux.domain.usecase.ClearHistoryUseCase
import com.bridgedeux.domain.usecase.DeleteHistoryUseCase

import com.bridgedeux.engine.mock.translation.MockTranslationRepository

import com.bridgedeux.feature.about.presentation.aboutViewModelFactory
import com.bridgedeux.feature.history.presentation.historyViewModelFactory
import com.bridgedeux.feature.settings.presentation.settingsViewModelFactory
import com.bridgedeux.feature.translation.presentation.translationViewModelFactory

class AppContainer(
    context: Context
) {

    private val settingsDataStore: DataStore<Preferences> =
        PreferenceDataStoreFactory.create(
            produceFile = {
                context.applicationContext.preferencesDataStoreFile(
                    "bridgedeux_settings"
                )
            }
        )

    private val settingsRepository =
        LocalSettingsRepository(
            dataStore = settingsDataStore
        )

    /*
     * -------------------------------------------------------------------------
     * Local Database
     * -------------------------------------------------------------------------
     *
     * The database belongs to the application-level composition root.
     * Feature modules must not create or own the Room database.
     */

    private val database: BridgeDeuxDatabase =
        Room.databaseBuilder(
            context.applicationContext,
            BridgeDeuxDatabase::class.java,
            "bridgedeux.db"
        )
            .build()

    /*
     * -------------------------------------------------------------------------
     * Translation
     * -------------------------------------------------------------------------
     *
     * The translation engine is still the mock implementation for now.
     * We will replace this with the real MarianMT ONNX implementation later.
     */

    private val translationRepository =
        MockTranslationRepository()

    private val translateTextUseCase =
        TranslateTextUseCase(
            translationRepository
        )

    /*
     * -------------------------------------------------------------------------
     * History
     * -------------------------------------------------------------------------
     *
     * History now uses the real Room-backed repository instead of the mock
     * repository.
     */

    private val historyRepository =
        LocalHistoryRepository(
            translationDao = database.translationDao()
        )

    private val getHistoryUseCase =
        GetHistoryUseCase(
            historyRepository
        )

    private val saveHistoryUseCase =
        SaveHistoryUseCase(
            historyRepository
        )

    private val deleteHistoryUseCase =
        DeleteHistoryUseCase(
            historyRepository
        )

    private val clearHistoryUseCase =
        ClearHistoryUseCase(
            historyRepository
        )

    /*
     * -------------------------------------------------------------------------
     * Settings / About
     * -------------------------------------------------------------------------
     */

    val settingsViewModelFactory =
        settingsViewModelFactory(
            settingsRepository = settingsRepository
        )

    val aboutViewModelFactory =
        aboutViewModelFactory()

    /*
     * -------------------------------------------------------------------------
     * Translation ViewModel
     * -------------------------------------------------------------------------
     */

    val translationViewModelFactory: ViewModelProvider.Factory =
        translationViewModelFactory(
            translateTextUseCase = translateTextUseCase,
            saveHistoryUseCase = saveHistoryUseCase
        )

    /*
     * -------------------------------------------------------------------------
     * History ViewModel
     * -------------------------------------------------------------------------
     */

    val historyViewModelFactory: ViewModelProvider.Factory =
        historyViewModelFactory(
            getHistoryUseCase = getHistoryUseCase,
            deleteHistoryUseCase = deleteHistoryUseCase,
            clearHistoryUseCase = clearHistoryUseCase
        )
}
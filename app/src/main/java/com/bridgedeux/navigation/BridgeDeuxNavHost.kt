package com.bridgedeux.navigation

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.bridgedeux.AppContainer
import com.bridgedeux.feature.history.presentation.HistoryRoute
import com.bridgedeux.feature.history.presentation.HistoryViewModel
import com.bridgedeux.feature.translation.presentation.TranslationRoute
import com.bridgedeux.feature.translation.presentation.TranslationViewModel
import com.bridgedeux.feature.settings.presentation.SettingsRoute
import com.bridgedeux.feature.settings.presentation.SettingsViewModel
import com.bridgedeux.feature.about.presentation.AboutRoute
import com.bridgedeux.feature.about.presentation.AboutViewModel

@Composable
fun BridgeDeuxNavHost(
    navController: NavHostController,
    innerPadding: PaddingValues,
    appContainer: AppContainer
) {

    NavHost(
        navController = navController,
        startDestination = Destination.Translate.route,
        modifier = Modifier.padding(innerPadding)
    ) {

        composable(Destination.Translate.route) {

            val translationViewModel: TranslationViewModel = viewModel(
//                factory = translationViewModelFactory(
//                    appContainer.translateTextUseCase
//                )
                factory = appContainer.translationViewModelFactory
            )

            TranslationRoute(
                viewModel = translationViewModel
            )
        }

        composable(Destination.History.route) {

            val historyViewModel: HistoryViewModel = viewModel(
//                factory = historyViewModelFactory(
//                    appContainer.getHistoryUseCase
//                )
                factory = appContainer.historyViewModelFactory
            )

            HistoryRoute(
                viewModel = historyViewModel
            )
        }

//        composable(Destination.Settings.route) {
//            Text("Settings (Coming Soon)")
//        }

//        composable(Destination.Settings.route) {
//            val settingsViewModel: SettingsViewModel = viewModel(
//                factory = appContainer.settingsViewModelFactory
//            )
//
//            SettingsRoute(
//                viewModel = settingsViewModel
//            )
//        }

        composable(Destination.Settings.route) {

            val settingsViewModel: SettingsViewModel = viewModel(
                factory = appContainer.settingsViewModelFactory
            )

            SettingsRoute(
                viewModel = settingsViewModel,
                onNavigateToAbout = {
                    navController.navigate(Destination.About.route)
                }
            )
        }

        composable(Destination.About.route) {
            val aboutViewModel: AboutViewModel = viewModel(
                factory = appContainer.aboutViewModelFactory
            )

            AboutRoute(
                viewModel = aboutViewModel
            )
        }
    }
}
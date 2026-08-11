package com.bridgedeux.navigation

sealed class Destination(
    val route: String,
    val label: String
) {

    data object Translate : Destination(
        route = "translate",
        label = "Translate"
    )

    data object History : Destination(
        route = "history",
        label = "History"
    )

    data object Settings : Destination(
        route = "settings",
        label = "Settings"
    )

    data object About : Destination(
        route = "about",
        label = "About"
    )

    companion object {

        val bottomBarDestinations = listOf(
            Translate,
            History,
            Settings
        )
    }
}
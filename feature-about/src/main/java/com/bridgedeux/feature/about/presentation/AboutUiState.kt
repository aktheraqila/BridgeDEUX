package com.bridgedeux.feature.about.presentation

/**
 * Represents the complete UI state for the About screen.
 *
 * This state contains only immutable display information.
 * It does not contain any business logic, Android framework
 * references, or mutable UI state.
 */
data class AboutUiState(

    // Application Information
    val appName: String = "",
    val version: String = "",

    // Research Information
    val researchTitle: String = "",
    val description: String = "",

    // Academic Information
    val supervisor: String = "",
    val department: String = "",
    val university: String = "",

    // Legal Information
    val license: String = ""
)
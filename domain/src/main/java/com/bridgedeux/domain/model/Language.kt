package com.bridgedeux.domain.model

enum class Language(
    val code: String
) {
    ENGLISH("en"),
    GERMAN("de");

    fun opposite(): Language {
        return when (this) {
            ENGLISH -> GERMAN
            GERMAN -> ENGLISH
        }
    }
}
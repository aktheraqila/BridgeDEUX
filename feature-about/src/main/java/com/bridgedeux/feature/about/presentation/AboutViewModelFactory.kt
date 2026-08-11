package com.bridgedeux.feature.about.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider

fun aboutViewModelFactory(): ViewModelProvider.Factory =
    object : ViewModelProvider.Factory {

        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(
            modelClass: Class<T>
        ): T {

            require(
                modelClass.isAssignableFrom(
                    AboutViewModel::class.java
                )
            ) {
                "Unknown ViewModel class: ${modelClass.name}"
            }

            return AboutViewModel() as T
        }
    }
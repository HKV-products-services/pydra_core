from .model.water_system import WaterSystem
from .profile.profile import Profile
from .settings.settings import Settings


class Location:
    """
    This is a class for one HRDLocation. This class contains the Settings,
    the WaterSystem mode (which is the model describing the statistics and
    loading) and a Profile. Which is a schematisation of the cross-section.
    """

    settings: Settings
    model: WaterSystem
    profile: Profile

    def __init__(self, settings: Settings, profile: Profile = None):
        """
        Initialize a Location object.

        Parameters
        ----------
        settings : Settings
            An instance of the Settings class that contains location-specific
            settings.
        profile : Profile
            An instance of the Profile class that provides a schematization of
            the cross-section.
        """
        self.settings = settings
        self.model = WaterSystem(settings)
        self.profile = profile

    def get_settings(self) -> Settings:
        """
        Return the Settings object.

        Returns
        -------
        Settings
            The Settings object for this Location
        """
        return self.settings

    def set_settings(self, settings: Settings) -> None:
        """
        Set the Settings for this location.

        Parameters
        ----------
        settings : Settings
            Settings object
        """
        # Check if the HRDLocation is still the same
        if self.settings.location != settings.location:
            raise ValueError(
                f"[ERROR] Cannot apply settings for location {settings.location} to location {self.settings.location}."
            )

        # Settings
        self.settings = settings

        # Statistics
        self.model = WaterSystem(settings)

    def get_model(self) -> WaterSystem:
        """
        Returns the WaterSystem model.

        Returns
        -------
        WaterSystem
            The WaterSystem model
        """
        return self.model

    def get_profile(self) -> Profile:
        """
        Return the Profile object

        Returns
        -------
        Loading
            The Loading object for this Location
        """
        return self.profile

    def has_profile(self) -> bool:
        """
        Returns whether of not a profile is assigned.

        Returns
        -------
        bool
            Profile assigned to Location
        """
        return isinstance(self.profile, Profile)

    def set_profile(self, profile: Profile = None) -> None:
        """
        Set the Profile for this location. By default, profile is none. Which
        deletes the profile from this location.

        Parameters
        ----------
        profile : Profile
            Profile object (default: None)
        """
        self.profile = profile

    def remove_profile(self) -> None:
        """
        Remove the Profile from the Location class.
        """
        self.profile = None

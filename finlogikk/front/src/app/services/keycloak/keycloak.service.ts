import { Injectable } from '@angular/core';
import Keycloak from "keycloak-js";
import {UserProfile} from "./user-profile";
import {Observable} from "rxjs";
import {HttpClient, HttpHeaders} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class KeycloakService {

  private _keycloak: Keycloak | undefined;
  private _profile: UserProfile | undefined;
  private _initialized = false;

  constructor(private http: HttpClient) { }

  get keycloak() {
    if (!this._keycloak) {
      this._keycloak = new Keycloak({
        url: 'http://localhost:9090',
        realm: 'finlogik',
        clientId: 'finlogik'
      });
    }
    return this._keycloak;
  }

  get profile(): UserProfile | undefined {
    return this._profile;
  }


  async init() {
    if (this._initialized) {
      console.log("Keycloak already initialized.");
      return;
    }

    const authenticated = await this.keycloak.init({
      onLoad: 'login-required',
    });

    if (authenticated) {
      this._profile = (await this.keycloak.loadUserProfile()) as UserProfile;
      this._profile.token = this.keycloak.token || '';
    }

    this._initialized = true;
  }

  login() {
    this.keycloak.login();
  }

  logout() {
    this.keycloak.logout({ redirectUri: window.location.origin });
  }

  getAllUsers(): Observable<any> {
    const token = this.keycloak.token;

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    });

    const keycloakUrl = 'http://localhost:9090/auth/admin/realms/finlogik/users';

    return this.http.get(keycloakUrl, { headers });
  }

}

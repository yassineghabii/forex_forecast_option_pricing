import { Component, OnDestroy, OnInit } from '@angular/core';

import {KeycloakService} from "../../services/keycloak/keycloak.service";
import {Router} from "@angular/router";

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent implements OnInit {

  isAuthenticated: boolean = false;
  profileName: string = '';

  constructor(
    public keycloakService: KeycloakService,
    private router: Router
  ) {}

  async ngOnInit() {
    this.isAuthenticated = await this.keycloakService.keycloak?.authenticated ?? false;
    if (this.isAuthenticated) {
      this.profileName = (await this.keycloakService.keycloak?.loadUserProfile())?.username ?? '';
    }
  }

  logout() {
    this.keycloakService.logout();
  }

  navigateToProfile() {
    const realm = 'finlogik'; // Replace with your actual realm
    const profileUrl = `http://localhost:9090/realms/${realm}/account`;
    window.open(profileUrl, '_blank');
  }

}

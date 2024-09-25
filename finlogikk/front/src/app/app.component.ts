  import { Component, OnInit, inject } from '@angular/core';
  import { NgxSpinnerService } from 'ngx-spinner';
  import { Observable, fromEvent, map } from 'rxjs';
  import { Router, NavigationEnd } from '@angular/router';
  import { DOCUMENT, ViewportScroller } from '@angular/common';
  import {SocialAuthService, SocialUser} from "@abacritt/angularx-social-login";
  // import {KeycloakService} from "./services/keycloak/keycloak.service";

  @Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css'],
  })
  export class AppComponent implements OnInit {
    title = 'front';
    isAuthenticated: boolean = false;
    profileName: string = '';
    constructor(
      private spinner: NgxSpinnerService,
      private router: Router,
      // private keycloakService: KeycloakService
  ) {}

    async ngOnInit() {

      setTimeout(() => {
        /** spinner ends after 5 seconds */
        this.spinner.hide();
      }, 1000);

      // // Initialize Keycloak
      // await this.keycloakService.init();
      //
      // // Check authentication status
      // this.isAuthenticated = this.keycloakService.keycloak.authenticated ?? false;
      // if (this.isAuthenticated) {
      //   this.profileName = (await this.keycloakService.keycloak.loadUserProfile())?.username ?? '';
      // }

    }

    private readonly document = inject(DOCUMENT);
    private readonly viewport = inject(ViewportScroller);

    readonly showScroll$: Observable<boolean> = fromEvent(
      this.document,
      'scroll'
    ).pipe(map(() => this.viewport.getScrollPosition()?.[1] > 0));

    onScrollToTop(): void {
      this.viewport.scrollToPosition([0, 0]);
    }

    shouldShowHeaderAndFooter(): boolean {
      const excludedRoutes = ['login', 'inscription', 'reset', 'forget'];
      const currentRoute = this.router.routerState.snapshot.url.split('/')[1];
      return !excludedRoutes.includes(currentRoute);
    }


  }
